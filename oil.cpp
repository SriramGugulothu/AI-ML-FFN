#include<iostream>
#include<vector>
#include<queue>
#include<climits>
using namespace std;
void fun(int index,int k,int n,vector<int>&companies,int &ans,int maxi,int mini){
    if(k == 1){
        int sum = 0;
        for(int i=index;i<n;i++){
            sum+=companies[i];
        } 
        maxi = max(maxi,sum);
        mini = min(mini,sum);
        cout<<maxi<<" "<<mini<<"\n";
        ans = min(ans,maxi-mini);
        return;
    }
    int sum = 0;
    int maxi2 = INT_MIN;
    int mini2 = INT_MAX;
    for(int i=index;i <= n-k;i++){
        sum += companies[i];
        maxi2 = max(sum,maxi);
        mini2 = min(sum,mini);
        fun(i+1,k-1,n,companies,ans,maxi2,mini2);
    }
}
int main(){
    int k;
    cin>>k;
    int n;
    cin>>n;
    vector<int>companies(n);
    for(int i=0;i<n;i++){
        cin>>companies[i];
    }
    int ans = INT_MAX;
    fun(0,k,n,companies,ans,INT_MIN,INT_MAX);
    cout<<ans;
    return 0;
}