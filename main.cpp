#include "LDA.h"

int main(int argc,char *argv[]){
    if(argc != 4){
        cout << "The number of argument is invalid." << endl;
        return(0);
    }
    int number_of_document = atoi(argv[1]);
    int start_topic = atoi(argv[2]);
    int correct_topic = atoi(argv[3]);
    run_FAB_LDA(number_of_document, start_topic, correct_topic);
}
