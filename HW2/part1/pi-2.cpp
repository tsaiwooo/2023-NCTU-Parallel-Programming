#include <iostream>
#include <ctime>
#include <cstdlib>
#include <string>
#include <iomanip>
#include <pthread.h>
#include <unistd.h>

using namespace std;

struct data{
    int numbers;
    int id;
};

int numbers;

void* Calculate(void *arg)
{
    // clock_t start,end;
    srand(time(NULL));
    unsigned int seed = pthread_self();

    int *res = new int(1);

    struct data *d;
    d = (struct data *)numbers;

    float min = -1;
    float max = 1;
    register long long int num_in_circle = 0;
    register float x,y,distance_squared;

    for(int i=0 ; i<numbers ; i++){
        x = (max-min) * rand_r(&seed) / (RAND_MAX+1.0) + min;
        y = (max-min) * rand_r(&seed) / (RAND_MAX+1.0) + min;


        distance_squared = x*x + y*y;
        if(distance_squared<=1){
            num_in_circle++;
        }
    }
    *(unsigned long long int *)arg = num_in_circle;
    // end = clock();
    // cout<<"time = "<<((double) (end - start))<<"\n";
    // res[0] = num_in_circle;
    // pthread_exit((void *)res);
    pthread_exit(NULL);
}

int main(int argc,char **argv)
{
    //time calculate
    // clock_t start,end;
    //get the data from cmd
    int thread;
    thread = stoi(argv[1]);
    long long int tosses;
    tosses = stoll(argv[2],nullptr);

    // struct data d1[thread];

    pthread_t t[thread];
    long long int res[thread];
    // long long int res[thread];
    // void *res[thread];
    long long int all = 0;
    numbers = tosses/thread;
    for(int i=0 ; i<thread ; i++){
        // d1[i].numbers = tosses/thread;
        // d1[i].id = i;
        pthread_create(&t[i],NULL,Calculate,(void *)res+i);
    }
    
    // start = clock();
    for(int i=0 ; i<thread ; i++){
        pthread_join(t[i],NULL);
        all += res[i];
    }
    // end = clock();
    // cout<<"end time = "<<((double)(end - start))<<"\n";


    float pi_estimate = 4*all / ((float) tosses);
    cout<<"estimate pi = "<<fixed<<setprecision(3)<<pi_estimate<<"\n";

}