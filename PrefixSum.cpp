/*white blue*/
#include<iostream>
#include<vector>
using namespace std;
void fun(vector<vector<int>> & prefixSum,int r,int c,int size,int &white,int &blue){
     
    int val = prefixSum[r+size-1][c+size-1] - prefixSum[r+size-1][c-1] - prefixSum[r-1][c+size-1] + prefixSum[r-1][c-1];
    if(val == size*size){
        cout<<r<<" "<<c<<"\n";
        blue+=1;
        return;
    }
    else if(val == 0){
        cout<<r<<" "<<c<<"\n";
        white+=1;
        return;
    }
    else{
        fun(prefixSum,r,c,size/2,white,blue);
        fun(prefixSum,r+size/2,c,size/2,white,blue);
        fun(prefixSum,r,c+size/2,size/2,white,blue);
        fun(prefixSum,r+size/2,c+size/2,size/2,white,blue);
    }
}
int main(){
    int t;
    cin>>t;
    int n;
    while(t-->0){
        cin>>n;
        vector<vector<int>>arr(n+1,vector<int>(n+1,0));
        for(int i=1;i<=n;i++){
            for(int j=1;j<=n;j++){
                cin>>arr[i][j];
            }
        }
        vector<vector<int>>prefixSum(n+1,vector<int>(n+1,0));
        for(int i=0;i<=n;i++){
            for(int j=0;j<=n;j++){
                if(i==0 || j==0){
                    prefixSum[i][j] = 0;
                }
                else{
                prefixSum[i][j] = prefixSum[i][j-1]+prefixSum[i-1][j]-prefixSum[i-1][j-1] + arr[i][j];
                }
            }
        }
        int white=0, blue = 0;
        fun(prefixSum,1,1,n,white,blue);
        cout<<white<<" "<< blue;
    }
    return 0;
}