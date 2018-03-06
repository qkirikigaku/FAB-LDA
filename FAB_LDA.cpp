//
// Created by 松谷太郎 on 2017/02/03.
//

#include "LDA.h"

void run_FAB_LDA(int number_of_document, int start_topic, int correct_topic){
  LDA temp_object(number_of_document, start_topic, correct_topic);
  temp_object.load_data();
  temp_object.run_FAB();
  temp_object.write_data();
}
