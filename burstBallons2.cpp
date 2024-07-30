/* Burst ballons -1 */
#include <iostream>
#include<vector>
#include<climits>
using namespace std;
int fun(vector<int>ballons,int i,int j,int count){
    if(i>j){
        return 0;
    }
    int maxi = INT_MIN;
    for(int k = i;k<=j;k++){
        int val = 0;
        if(count==0){
            val = ballons[k];
        }
        else{
            val = ballons[i-1]*ballons[j+1];
        }
        int temp = fun(ballons,i,k-1,count+1) + fun(ballons,k+1,j,count+1);
        maxi = max(maxi,temp+val);
    }
    return maxi;
}
int main() {
    int n;
    cin>>n;
    vector<int>ballons(n+2);
    ballons[0] = 1;
    ballons[n+1] =1;
    for(int i=1;i<=n;i++){
        cin>>ballons[i];
    }    
    int count = 0;
   cout<<fun(ballons,1,n,count);
}
