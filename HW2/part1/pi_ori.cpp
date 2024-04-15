#include <iostream>
#include <ctime>
#include <cstdlib>
#include <string>
#include <iomanip>

using namespace std;

// double min = -1;
// double max = 1;

int main(int argc,char **argv)
{
    srand(time(NULL));

    //get the data from cmd
    int thread;
    thread = stoi(argv[1]);
    long long int tosses;
    tosses = stoll(argv[2],nullptr);

    double min = -1;
    double max = 1;
    long long int num_in_circle = 0;


    for(int toss=0 ; toss < tosses ; toss++){
        double x = (max-min) * rand() / (RAND_MAX+1.0) + min;
        double y = (max-min) * rand() / (RAND_MAX+1.0) + min;
        
        double distance_squared = x*x + y*y;
        if(distance_squared<=1){
            num_in_circle++;
        }
    }

    double pi_estimate = 4*num_in_circle / ((double) tosses);
    cout<<"estimate pi = "<<fixed<<setprecision(9)<<pi_estimate<<"\n";
}