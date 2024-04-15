#include <iostream>
#include <ctime>
#include <cstdlib>
#include <string>
#include <iomanip>
#include <pthread.h>
#include <unistd.h>
#include <stdio.h>

using namespace std;

struct data{
    int numbers;
    int id;
    int t_th;
};

void* Calculate(void *numbers)
{
    // clock_t start,end;
    srand(10000);
    // unsigned int seed = pthread_self();
    unsigned int seed = 2023;

    int *res = new int(1);

    struct data *d;
    d = (struct data *)numbers;

    float min = -1;
    float max = 1;
    register long long int num_in_circle = 0;
    register float x,y,distance_squared;

    // for(int i=0 ; i<d->numbers ; i++){
    for(int i=d->id;i<d->numbers;i+=d->t_th){
        // x = (max-min) * rand_r(&seed) / (RAND_MAX) + min;
        // y = (max-min) * rand_r(&seed) / (RAND_MAX) + min;
        x = (double)rand_r(&seed)/RAND_MAX ;
        y = (double)rand_r(&seed)/RAND_MAX ;


        distance_squared = x*x + y*y;
        if(distance_squared<=1){
            num_in_circle++;
        }
    }
    // end = clock();
    // cout<<"time = "<<((double) (end - start))<<"\n";
    res[0] = num_in_circle;
    pthread_exit((void *)res);
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

    struct data d1[thread];

    pthread_t t[thread];
    // long long int res[thread];
    void *res[thread-1];
    register long long int all = 0;
    for(int i=0 ; i<thread-1 ; i++){
        // d1[i].numbers = tosses/thread;
        d1[i].numbers = tosses;
        d1[i].id = i;
        d1[i].t_th = thread;
        pthread_create(&t[i],NULL,Calculate,(void *)&d1[i]);
    }


    float x,y,distance_squared;
    unsigned int seed = 2023;

    // for(int i=0;i<tosses/thread;i++){
    for(int i=thread-1 ; i<tosses ; i+=thread){
        x = (double)rand_r(&seed)/RAND_MAX ;
        y = (double)rand_r(&seed)/RAND_MAX ;


        distance_squared = x*x + y*y;
        if(distance_squared<=1){
            all++;
        }
    }
    
    // start = clock();
    for(int i=0 ; i<thread-1 ; i++){
        pthread_join(t[i],&res[i]);
        all += *(int *)res[i];
    }
    // end = clock();
    // cout<<"end time = "<<((double)(end - start))<<"\n";


    float pi_estimate = 4*all / ((float) tosses);
    cout<<pi_estimate<<"\n";
    // printf("%f\n", pi_estimate);

}