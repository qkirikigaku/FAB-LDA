//
// Created by 松谷太郎 on 2017/02/03.
//

#ifndef LDA_LDA_H
#define LDA_LDA_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/gamma.hpp>

using namespace std;

class LDA {
private:
    int number_of_topic;
    int number_of_word;
    int number_of_document;
    int iter;
    //document[D:Dth document][I:Ith word]
    vector<vector<int> > train_document;
    vector<vector<int> > test_document;

    vector<vector<vector<double> > > log_responsibility;
    //q(z)　= responsibility[D:Dth document][I:Ith word][K:Kth topic]

    vector<vector<double> > xi_theta;
    //xi_theta[D:Dth document][K::Kth topic]

    vector<vector<double> > xi_phi;
    //xi_phi[K:Kth topic][V:Vth word]

    vector<double> alpha;
    //alpha[K:Kth topic]

    vector<double> beta;
    //beta[V:Vth word]

    vector<vector<double> > log_sum_res_for_i;
    //E[n_{dk}]

    vector<vector<double> > log_sum_res_for_di;
    //E[n_{kv}]

    vector<double> FIC;

    double temp_variational_lower_bound;
    double old_variational_lower_bound;
    //double perplexity;

public:
    LDA(int x, int y) {
        number_of_document = x;
	number_of_topic = y;
    }

    void run_VB() {
        initialize();
        old_variational_lower_bound = -10000000000000;
        Update_parameter();
        for (int i = 0; i < 1000; i++) {
            cout << "iter:" << i << endl;
            Update_log_responsibility();
            Update_parameter();
            Update_hyperparameter();
            calc_variational_lower_bound();
	    if(i == 369){
	      write_data();
	    }
            if (fabs(temp_variational_lower_bound - old_variational_lower_bound) < 0.1) {
                show_ELBO();
                cout << endl;
		show_alpha();
                break;
            } else {
                show_ELBO();
		shrinkage();
            }
            cout << endl;
        }
        //calc_perplexity();
    }

    void initialize(){
        int d, i, k, v;
	iter = 0;
	FIC.reserve(2000);

        log_responsibility.resize(number_of_document);
        for (d = 0; d < number_of_document; d++) {
            log_responsibility[d].resize(train_document[d].size());
            for (i = 0; i < train_document[d].size(); i++) {
                log_responsibility[d][i].resize(number_of_topic, (double) 1 / number_of_topic);
            }
        }

        alpha.resize(number_of_topic);
        for (k = 0; k < number_of_topic; k++) {
	  alpha[k] = (double) 0.01 * (k + 1);
        }

        beta.resize(number_of_word);
        for (v = 0; v < number_of_word; v++) {
            beta[v] = (double)(v%30 + 1)*0.0001;
	    //beta[v] = 0.1;
        }

        xi_theta.resize(number_of_document);
        for (d = 0; d < number_of_document; d++) {
            xi_theta[d].resize(number_of_topic, 0);
        }

        xi_phi.resize(number_of_topic);
        for (k = 0; k < number_of_topic; k++) {
            xi_phi[k].resize(number_of_word, 0);
        }

        log_sum_res_for_i.resize(number_of_document);
        for (d = 0; d < number_of_document; d++) {
            log_sum_res_for_i[d].resize(number_of_topic, 0);
        }
	calc_n_dk();

        log_sum_res_for_di.resize(number_of_topic);
        for (k = 0; k < number_of_topic; k++) {
            log_sum_res_for_di[k].resize(number_of_word, -730);
        }
	calc_n_kv();
    };

    void Update_log_responsibility() {
        int d, v, i, k;

        vector<double> sum_xi_theta;
        sum_xi_theta.resize(number_of_document, 0);
        for (d = 0; d < number_of_document; d++) {
            for (k = 0; k < number_of_topic; k++) {
                sum_xi_theta[d] += xi_theta[d][k];
            }
        }

        vector<double> sum_xi_phi;
        sum_xi_phi.resize(number_of_topic, 0);
        for (k = 0; k < number_of_topic; k++) {
            for (v = 0; v < number_of_word; v++) {
                sum_xi_phi[k] += xi_phi[k][v];
            }
        }

        vector<vector<vector<double> > > old_log_responsibility = log_responsibility;

	vector<double> sum_res_by_k;
	calc_n_dk();
	sum_res_by_k.resize(number_of_topic,0);
	for(k=0; k<number_of_topic; k++){
	  for(d=0; d<number_of_document; d++){
	    sum_res_by_k[k] += exp(log_sum_res_for_i[d][k]);
	  }
	}

	cout << "sum_res_by_k[k] : " << endl;
	for(k=0;k<number_of_topic; k++){
	  cout << to_string(sum_res_by_k[k]) << ", ";
	}
	cout << endl;

        for (d = 0; d < number_of_document; d++) {
            for (i = 0; i < train_document[d].size(); i++) {
                for (k = 0; k < number_of_topic; k++) {
                    log_responsibility[d][i][k] =
                            boost::math::digamma(xi_phi[k][train_document[d][i]]) + boost::math::digamma(xi_theta[d][k])
		      - boost::math::digamma(sum_xi_phi[k]) - boost::math::digamma(sum_xi_theta[d])
		      - number_of_word/(2*sum_res_by_k[k]);
                }
            }
        }

        vector<vector<vector<double> > > temp_log_responsibility = log_responsibility;

        for (d = 0; d < number_of_document; d++) {
            for (i = 0; i < train_document[d].size(); i++) {
                int max_k = 0;
                for (k = 1; k < number_of_topic; k++) {
                    if (log_responsibility[d][i][k] > log_responsibility[d][i][max_k]) {
                        max_k = k;
                    }
                }
                double sum_exponential = 0;
                for (k = 0; k < number_of_topic; k++) {
                    sum_exponential += exp(log_responsibility[d][i][k] - log_responsibility[d][i][max_k]);
                }
                for (k = 0; k < number_of_topic; k++) {
                    log_responsibility[d][i][k] -= (temp_log_responsibility[d][i][max_k] + log(sum_exponential));
                }
            }
        }
    };

    void Update_parameter() {
        int d, v, i, k;
       
        //Update xi_theta
        calc_n_dk();
        for (d = 0; d < number_of_document; d++) {
            for (k = 0; k < number_of_topic; k++) {
                xi_theta[d][k] = exp(log_sum_res_for_i[d][k]) + alpha[k];
            }
        }
	
        //Update xi_phi
        calc_n_kv();
        for (k = 0; k < number_of_topic; k++) {
            for (v = 0; v < number_of_word; v++) {
                xi_phi[k][v] = exp(log_sum_res_for_di[k][v]) + beta[v];
            }
        }
    };

    void Update_hyperparameter() {
        int d, v, i, k;

        //Update alpha
        vector<double> new_numerator_alpha;
        new_numerator_alpha.resize(number_of_topic, 0);

        calc_n_dk();

        for (k = 0; k < number_of_topic; k++) {
            for (d = 0; d < number_of_document; d++) {
                new_numerator_alpha[k] += (boost::math::digamma(exp(log_sum_res_for_i[d][k]) + alpha[k]) -
                                           boost::math::digamma(alpha[k])) * alpha[k];
            }
        }
        double new_denominator_alpha = 0, sum_alpha = 0;
        for (k = 0; k < number_of_topic; k++) {
            sum_alpha += alpha[k];
        }
        for (d = 0; d < number_of_document; d++) {
            new_denominator_alpha +=
                    boost::math::digamma(train_document[d].size() + sum_alpha) - boost::math::digamma(sum_alpha);
        }

        int count = 0;
        for (k = 0; k < number_of_topic; k++) {
            if (new_numerator_alpha[k] == 0) {
                count++;
            }
        }
        if (count == 0) {
            for (k = 0; k < number_of_topic; k++) {
                alpha[k] = new_numerator_alpha[k] / new_denominator_alpha;
            }
        }

        //Update beta

         //Update beta for each word
        vector<double> new_numerator_beta;
        vector<double> new_denominator_beta;
        new_numerator_beta.resize(number_of_word,0);
        new_denominator_beta.resize(number_of_word,0);

        calc_n_kv();
        for(v=0; v<number_of_word; v++){
            for(k=0; k<number_of_topic; k++){
                new_numerator_beta[v] += (boost::math::digamma(exp(log_sum_res_for_di[k][v])+beta[v])-boost::math::digamma(beta[v]))*beta[v];
            }
        }

        vector<double> sum_res_for_div;
        double sum_beta = 0;
        sum_res_for_div.resize(number_of_topic,0);
        for(k=0; k<number_of_topic; k++){
            for(v=0; v<number_of_word; v++){
                sum_res_for_div[k] += exp(log_sum_res_for_di[k][v])+beta[v];
            }
        }
        for(v=0; v<number_of_word; v++){
            sum_beta += beta[v];
        }
        for(v=0; v<number_of_word; v++){
            for(k=0; k<number_of_topic; k++){
                new_denominator_beta[v] += boost::math::digamma(sum_res_for_div[k])-boost::math::digamma(sum_beta);
            }
        }
        for(v=0; v<number_of_word; v++){
            beta[v] = new_numerator_beta[v] / new_denominator_beta[v];
        }

        // Update single beta
        /*
        double identity_numerator_beta = 0;
        double identity_denominator_beta = 0;
        calc_n_kv();
        for (k = 0; k < number_of_topic; k++) {
            for (v = 0; v < number_of_word; v++) {
                identity_numerator_beta += (boost::math::digamma(exp(log_sum_res_for_di[k][v]) + beta[0]) -
                                            boost::math::digamma(beta[0])) * beta[0];
            }
        }

        vector<double> log_sum_res_for_div, sum_res_for_div;
        log_sum_res_for_div.resize(number_of_topic, 0);
        sum_res_for_div.resize(number_of_topic, 0);
        for (k = 0; k < number_of_topic; k++) {
            int max_v = 0;
            for (v = 1; v < number_of_word; v++) {
                if (log_sum_res_for_di[k][v] > log_sum_res_for_di[k][max_v]) {
                    max_v = v;
                }
            }
            for (v = 0; v < number_of_word; v++) {
                sum_res_for_div[k] += exp(log_sum_res_for_di[k][v] - log_sum_res_for_di[k][max_v]);
            }
            log_sum_res_for_div[k] = log(sum_res_for_div[k]) + log_sum_res_for_di[k][max_v];
        }

        for (k = 0; k < number_of_topic; k++) {
            identity_denominator_beta +=
                    boost::math::digamma(exp(log_sum_res_for_div[k]) + (double) number_of_word * beta[0])
                    - boost::math::digamma((double) number_of_word * beta[0]);
        }

        for (v = 0; v < number_of_word; v++) {
            beta[v] = (identity_numerator_beta / identity_denominator_beta) / (double) number_of_word;
        }
         */
    };

    void calc_variational_lower_bound() {
        int d, v, i, k;
        double first_component = 0, second_component = 0, third_component = 0, fourth_component = 0, fifth_component = 0;
	double FIC_component = 0;

        //calc_first_component
        double Vbeta = 0;
        double sum_lgamma_beta = 0;
        for(v=0; v<number_of_word; v++){
            Vbeta += beta[v];
            sum_lgamma_beta += boost::math::lgamma(beta[v]);
        }
        for (k = 0; k < number_of_topic; k++) {
            double sum_xi_phi_for_v = 0, log_pi_gamma_xi_phi_for_v = 0;
            for (v = 0; v < number_of_word; v++) {
                sum_xi_phi_for_v += xi_phi[k][v];
                log_pi_gamma_xi_phi_for_v += boost::math::lgamma(xi_phi[k][v]);
            }
            first_component += boost::math::lgamma(Vbeta)
                               - sum_lgamma_beta
                               - boost::math::lgamma(sum_xi_phi_for_v)
                               + log_pi_gamma_xi_phi_for_v;
        }

        //calc_second_component
        calc_n_kv();
        for (k = 0; k < number_of_topic; k++) {
            double sum_xi_phi_for_v = 0;
            for (v = 0; v < number_of_word; v++) {
                sum_xi_phi_for_v += xi_phi[k][v];
            }
            for (v = 0; v < number_of_word; v++) {
                second_component += (exp(log_sum_res_for_di[k][v]) + beta[v] - xi_phi[k][v]) *
                                    (boost::math::digamma(xi_phi[k][v]) - boost::math::digamma(sum_xi_phi_for_v));
            }
        }

        //calc_third_component
        double sum_alpha = 0, log_sum_gamma_alpha = 0;
        for (k = 0; k < number_of_topic; k++) {
            sum_alpha += alpha[k];
            log_sum_gamma_alpha += boost::math::lgamma(alpha[k]);
        }
        for (d = 0; d < number_of_document; d++) {
            double sum_xi_theta_for_k = 0, log_sum_gamma_xi_theta_for_k = 0;
            for (k = 0; k < number_of_topic; k++) {
                sum_xi_theta_for_k += xi_theta[d][k];
                log_sum_gamma_xi_theta_for_k += boost::math::lgamma(xi_theta[d][k]);
            }
            third_component += boost::math::lgamma(sum_alpha)
                               - log_sum_gamma_alpha
                               - boost::math::lgamma(sum_xi_theta_for_k)
                               + log_sum_gamma_xi_theta_for_k;
        }

        //calc_fourth_component
        calc_n_dk();
        for (d = 0; d < number_of_document; d++) {
            double sum_xi_theta_for_k = 0;
            for (k = 0; k < number_of_topic; k++) {
                sum_xi_theta_for_k += xi_theta[d][k];
            }
            for (k = 0; k < number_of_topic; k++) {
                fourth_component += (exp(log_sum_res_for_i[d][k]) + alpha[k] - xi_theta[d][k]) *
                                    (boost::math::digamma(xi_theta[d][k]) - boost::math::digamma(sum_xi_theta_for_k));
            }
        }

        //calc_fifth_component
        for (d = 0; d < number_of_document; d++) {
            for (i = 0; i < train_document[d].size(); i++) {
                for (k = 0; k < number_of_topic; k++) {
                    fifth_component += exp(log_responsibility[d][i][k]) * log_responsibility[d][i][k];
                }
            }
        }
	
	//FIC_component
	vector<double> res_by_k;
	res_by_k.resize(number_of_topic,0);
	for(k=0; k<number_of_topic; k++){
	  for(d=0; d<number_of_document; d++){
	    res_by_k[k] += exp(log_sum_res_for_i[d][k]);
	  }
	  FIC_component += (double)number_of_word * log(res_by_k[k])/2;
	}
	double sum_log_word = 0;
	for(d=0;d<number_of_document;d++){
	  sum_log_word += (double)number_of_topic * log(train_document[d].size())/2;
	}
	FIC_component += sum_log_word;

        /*if (isinf(first_component)) {
            cout << "first" << endl;
        }
        if (isinf(second_component)) {
            cout << "second" << endl;
        }
        if (isinf(third_component)) {
            cout << "third" << endl;
        }
        if (isinf(fourth_component)) {
            cout << "fourth" << endl;
        }
        if (isinf(fifth_component)) {
            cout << "fifth" << endl;
	    }*/

        temp_variational_lower_bound =
                first_component + second_component + third_component + fourth_component - fifth_component - FIC_component;
    
	FIC.push_back(temp_variational_lower_bound);
	iter += 1;
    }
    /*
    void calc_perplexity(){
        int d,v,i,k;
        perplexity = 0;
        calc_n_dk();
        calc_n_kv();
        double log_sum_likelihood = 0;
        double sum_alpha = 0;
        for(k=0; k<number_of_topic; k++){
            sum_alpha += alpha[k];
        }
        for(d=0; d<number_of_document; d++){
            for(i=0; i<test_document[d].size(); i++){
                double sum_likelihood = 0;
                for(k=0; k<number_of_topic; k++){
                    double likelihood_denominator = 0;
                    for(v=0; v<number_of_word; v++){
                        likelihood_denominator += exp(log_sum_res_for_di[k][v]) + beta[v];
                    }
                    likelihood_denominator *= (train_document[d].size()+sum_alpha);
                    sum_likelihood += (exp(log_sum_res_for_di[k][test_document[d][i]]) + beta[v])*(exp(log_sum_res_for_i[d][k])+alpha[k])
                            /likelihood_denominator;
                }
                log_sum_likelihood += log(sum_likelihood);
            }
        }
        double sum_number_test = 0;
        for(d=0; d<number_of_document; d++){
            sum_number_test += test_document[d].size();
        }
        perplexity = exp(-log_sum_likelihood/sum_number_test);
        cout << "perplexity: " << perplexity << endl;
    }*/

    void calc_n_dk() {
        int d, v, i, k;
        vector<vector<double> > sum_res_for_i;
        sum_res_for_i.resize(number_of_document);
        for (d = 0; d < number_of_document; d++) {
            sum_res_for_i[d].resize(number_of_topic, 0);
        }
        for (d = 0; d < number_of_document; d++) {
            for (k = 0; k < number_of_topic; k++) {
                int max_i = 0;
                for (i = 1; i < train_document[d].size(); i++) {
                    if (log_responsibility[d][i][k] > log_responsibility[d][max_i][k]) {
                        max_i = i;
                    }
                }
                for (i = 0; i < train_document[d].size(); i++) {
                    sum_res_for_i[d][k] += exp(log_responsibility[d][i][k] - log_responsibility[d][max_i][k]);
                }
                log_sum_res_for_i[d][k] = log(sum_res_for_i[d][k]) + log_responsibility[d][max_i][k];
            }
        }
    }

    void calc_n_kv() {
        int d, v, i, k;
        vector<vector<double> > sum_res_for_di;
        vector<vector<vector<int> > > max_di;
        sum_res_for_di.resize(number_of_topic);
        max_di.resize(number_of_topic);
        for (k = 0; k < number_of_topic; k++) {
            sum_res_for_di[k].resize(number_of_word, 0);
            max_di[k].resize(number_of_word);
            for (v = 0; v < number_of_word; v++) {
                max_di[k][v].resize(2, 100000);
            }
        }

        for (d = 0; d < number_of_document; d++) {
            for (i = 0; i < train_document[d].size(); i++) {
                if (max_di[0][train_document[d][i]][0] == 100000) {
                    for (k = 0; k < number_of_topic; k++) {
                        max_di[k][train_document[d][i]][0] = d;
                        max_di[k][train_document[d][i]][1] = i;
                    }
                } else {
                    for (k = 0; k < number_of_topic; k++) {
                        if (log_responsibility[d][i][k]
                            >
                            log_responsibility[max_di[k][train_document[d][i]][0]][max_di[k][train_document[d][i]][1]][k]) {
                            max_di[k][train_document[d][i]][0] = d;
                            max_di[k][train_document[d][i]][1] = i;
                        }
                    }
                }
            }
        }
        for (v = 0; v < number_of_word; v++) {
            for (d = 0; d < number_of_document; d++) {
                for (i = 0; i < train_document[d].size(); i++) {
                    if (train_document[d][i] == v) {
                        for (k = 0; k < number_of_topic; k++) {
                            sum_res_for_di[k][v]
                                    += exp(log_responsibility[d][i][k] -
                                           log_responsibility[max_di[k][v][0]][max_di[k][v][1]][k]);
                        }
                    }
                }
            }
        }
        for (k = 0; k < number_of_topic; k++) {
            for (v = 0; v < number_of_word; v++) {
                log_sum_res_for_di[k][v] =
                        log(sum_res_for_di[k][v]) + log_responsibility[max_di[k][v][0]][max_di[k][v][1]][k];
            }
        }
    }

    void shrinkage(){
      calc_n_dk();
      int d,i,k,v;
      vector<double> sum_res_by_k;
      sum_res_by_k.resize(number_of_topic,0);
      for(k=0; k<number_of_topic; k++){
	for(d=0; d<number_of_document; d++){
	  sum_res_by_k[k] += exp(log_sum_res_for_i[d][k]);
	}
      }
      int all_word_number = 0;
      for(d=0;d<number_of_document;d++){
	all_word_number += train_document[d].size();
      }
      vector<int> erase_topic;
      int erase_number = 0;
      erase_topic.resize(number_of_topic);
      for(k=0;k<number_of_topic;k++){
	if(sum_res_by_k[k] <= 0.01 * double(all_word_number)){
	  erase_topic[k] = 1;
	  erase_number += 1;
	}
	else{
	  erase_topic[k] = 0;
	}
      }

      int old_number_of_topic = number_of_topic;
      number_of_topic -= erase_number;
      double flag = 1000000.0;
                                                                                                                 
      for(k=0;k<old_number_of_topic;k++){
	if(erase_topic[k] == 1){
	  for(d=0;d<number_of_document;d++){
	    for(i=0;i<train_document[d].size();i++){
	      log_responsibility[d][i][k] = flag;
	    }
	  }
	  for(d=0;d<number_of_document;d++){
	    xi_theta[d][k] = flag;
	    log_sum_res_for_i[d][k] = flag;
	  }
	  for(v=0;v<number_of_word;v++){
	    xi_phi[k][v] = flag;
	    log_sum_res_for_di[k][v] = flag;
	  }
	  alpha[k] = flag;
	}
      }
      
      for(d=0;d<number_of_document;d++){
	for(i=0;i<train_document[d].size();i++){
	  k=0;
	  while(1){
	    if(k >= log_responsibility[d][i].size())break;
	    else{
	      if(log_responsibility[d][i][k] == flag){
		log_responsibility[d][i].erase(log_responsibility[d][i].begin() + k);
	      }
	      else{
		k += 1;
	      }
	    }
	  }
	}
      }

      for(d=0;d<number_of_document;d++){
	k=0;
	while(1){
	  if(k >= xi_theta[d].size())break;
	  else{
	    if(xi_theta[d][k] == flag){
	      xi_theta[d].erase(xi_theta[d].begin() + k);
	    }
	    else{
	      k += 1;
	    }
	  }
	}
	k=0;
	while(1){
	  if(k >= log_sum_res_for_i[d].size())break;
	  else{
	    if(log_sum_res_for_i[d][k] == flag){
	      log_sum_res_for_i[d].erase(log_sum_res_for_i[d].begin() + k);
	    }
	    else{
	      k += 1;
	    }
	  }
	}
      }

      k = 0;
      while(1){
	if(k >= xi_phi.size())break;
	else{
	  if(xi_phi[k][0] == flag){
	    xi_phi.erase(xi_phi.begin() + k);
	  }
	  else{
	    k += 1;
	  }
	}
      }
      k = 0;
      while(1){
	if(k >= log_sum_res_for_di.size())break;
	else{
	  if(log_sum_res_for_di[k][0] == flag){
	    log_sum_res_for_di.erase(log_sum_res_for_di.begin() + k);
	  }
	  else{
	    k += 1;
	  }
	}
      }
      k = 0;
      while(1){
	if(k >= alpha.size())break;
	else{
	  if(alpha[k] == flag){
	    alpha.erase(alpha.begin() + k);
	  }
	  else{
	    k += 1;
	  }
	}
      }


      vector<vector<vector<double> > > old_res(log_responsibility);
      //q(z)　= responsibility[D:Dth document][I:Ith word][K:Kth topic]

      cout << "number_of_topic : " << to_string(number_of_topic) << endl;
      for(d=0;d<number_of_document;d++){
	for(i=0;i<train_document[d].size();i++){
	  int max_k = 0;
	  double sum_res = 0;
	  for(k=0;k<number_of_topic;k++){
	    if(old_res[d][i][k] >= old_res[d][i][max_k]){
	      max_k = k;
	    }
	  }
	  for(k=0;k<number_of_topic;k++){
	    sum_res += exp(old_res[d][i][k] - old_res[d][i][max_k]);
	  }
	  for(k=0;k<number_of_topic;k++){
	    log_responsibility[d][i][k] -= (old_res[d][i][max_k] + log(sum_res));
	  }
	}
      }

      Update_parameter();

    }

    void load_data() {
        ifstream ifs;
        string input_file_name = "data/sample" + to_string(number_of_document) +".txt";
        ifs.open(input_file_name.c_str(), ios::in);
        if (!ifs) {
            cout << "Cannnot open " + input_file_name << endl;
            exit(1);
        }
        char buf[1000000];
        char *temp;
        vector<vector<int> > raw_document;
        vector<int> words_number;
        ifs.getline(buf, 1000000);
        temp = strtok(buf, " ");
        number_of_document = atoi(temp);
        raw_document.resize(number_of_document);
        words_number.resize(number_of_document,0);
        train_document.resize(number_of_document);
        test_document.resize(number_of_document);
        temp = strtok(NULL, " ");
        number_of_word = atoi(temp);
        int temp_word_number;

        for (int d = 0; d < number_of_document; d++) {
            ifs.getline(buf, 1000000);
            for (int v = 0; v < number_of_word; v++) {
                if (v == 0) temp_word_number = atoi(strtok(buf, " "));
                else temp_word_number = atoi(strtok(NULL, " "));
                for(int i = 0; i < temp_word_number; i++){
                    raw_document[d].push_back(v);
                    words_number[d]++;
                }
            }
        }
	
        for (int d = 0; d < number_of_document; d++){
            int count = 0;
	    train_document[d].resize(words_number[d]);
            for(int i = 0; i < words_number[d]; i++){
	      train_document[d][i] = raw_document[d][i];
	        /*if(count % 10 == 5){
                    test_document[d].push_back(raw_document[d][i]);
                }
                else{
                    train_document[d].push_back(raw_document[d][i]);
		}*/
                count ++;
            }
        }
        /*
        for(int d = 0; d < number_of_document; d++){
            random_device rnd;
            mt19937 mt(rnd());
            uniform_int_distribution<> rand_number(0,words_number[d]-1);
            vector<int> test_number;
            int number = (int)words_number[d]/10;
            test_number.resize(number);
            int count = 0;
            while(count != number){
                test_number[count] = rand_number(mt);
                int flag = 0;
                for(int c = 0; c < count; c++){
                    if(test_number[c] == test_number[count]){
                        flag = 1;
                    }
                }
                if(flag == 0)count++;
            }
            for(int i = 0; i < words_number[d]; i++){
                int flag = 0;
                int co = 0;
                while(co != words_number[d]){
                    if(i == test_number[co]){
                        flag = 1;
                    }
                    co++;
                }
                if(flag == 0)train_document[d].push_back(raw_document[d][i]);
                else test_document[d].push_back(raw_document[d][i]);
            }
        }
         */
        ifs.close();
    }

    void write_data() {
        ofstream ofs;
        string output_file_name = "result/sample" + to_string(number_of_document) +  "/result_k";
        if(number_of_topic < 10){
            output_file_name += "0" + to_string(number_of_topic) + ".txt";
        }
        else{
            output_file_name += to_string(number_of_topic) + ".txt";
        }
        ofs.open(output_file_name, ios::out);
        ofs << to_string(temp_variational_lower_bound) << "\n";
        ofs << "0" << "\n";
        int d, k, v;
        vector<vector<double> > Enkv;
        vector<double> sum_output;
        Enkv.resize(number_of_topic);
        sum_output.resize(number_of_topic, 0);
        for (k = 0; k < number_of_topic; k++) {
            Enkv[k].resize(number_of_word);
            for (v = 0; v < number_of_word; v++){
                sum_output[k] += exp(log_sum_res_for_di[k][v]);
            }
            for (v = 0; v < number_of_word; v++) {
                Enkv[k][v] = exp(log_sum_res_for_di[k][v]) / sum_output[k];
                ofs << to_string(Enkv[k][v]) << " ";
            }
            ofs << "\n";
        }
	vector<vector<double> > Endk;
	vector<double> sum_output_a;
	Endk.resize(number_of_document);
	sum_output_a.resize(number_of_document);
	for(d = 0; d < number_of_document; d++){
	  Endk[d].resize(number_of_topic, 0);
	  for(k = 0; k < number_of_topic; k++){
	    sum_output_a[d] += exp(log_sum_res_for_i[d][k]);
	  }
	  for(k = 0; k < number_of_topic; k++){
	    Endk[d][k] = exp(log_sum_res_for_i[d][k]) / sum_output_a[d];
	    ofs << to_string(Endk[d][k]) << " ";
	  }
	  ofs << "\n";
	}
	for (k = 0; k < number_of_topic; k++){
	  ofs << alpha[k] << " ";
	}
	ofs << "\n";
        ofs.close();
	ofstream FIC_out;
	output_file_name = "result/sample" + to_string(number_of_document) +  "/FIC_k";
        if(number_of_topic < 10){
	  output_file_name += "0" + to_string(number_of_topic) + ".txt";
        }
        else{
	  output_file_name += to_string(number_of_topic) + ".txt";
        }
        FIC_out.open(output_file_name, ios::out);
	for(k=0;k<iter;k++){
	  FIC_out << to_string(FIC[k]) << "\n";
	}
	FIC_out.close();
    }

    void show_alpha() {
        int k;
        cout << "alpha:" << endl;
        for (k = 0; k < number_of_topic; k++) {
            cout << alpha[k] << " ";
        }
        cout << endl;
    }

    void show_beta() {
        int v;
        cout << "beta:" << endl;
        for (v = 0; v < number_of_word; v++) {
            cout << beta[v] << " ";
        }
        cout << endl;
    }

    void show_xi_theta() {
        int d, k;
        cout << "xi_theta:" << endl;
        for (d = 0; d < number_of_document; d++) {
            for (k = 0; k < number_of_topic; k++) {
                cout << xi_theta[d][k] << " ";
            }
            cout << endl;
        }
    }

    void show_res() {
        cout << "q[1][5]:" << endl;
        for (int k = 0; k < number_of_topic; k++) {
            cout << exp(log_responsibility[0][4][k]) << ", ";
        }
        cout << endl;
    }

    void show_document() {
        int d, i;
        cout << "train_document[0]:" << endl;
        for (i = 0; i < train_document[0].size(); i++) {
            cout << train_document[0][i] << " ";
        }
        cout << endl;
    }

    void show_ELBO() {
        calc_variational_lower_bound();
        cout << "ELBO: " << temp_variational_lower_bound << endl;
        cout << "Improvement point: " << temp_variational_lower_bound - old_variational_lower_bound << endl;
        old_variational_lower_bound = temp_variational_lower_bound;
    }
};

void run_VB_LDA(int number_of_document, int number_of_topic);

#endif //LDA_LDA_H
