#include<iostream>
#include<vector>
#include<queue>
#include<climits>
using namespace std;
int dis(int r1,int c1,int r2, int c2){
    return abs(r1-r2)+abs(c1-c2);
}
int fun(int i,int mask,vector<int>&r,vector<int>&c){
    if(mask == (1<<i)|3 ){
        return dis(r[7],c[7],r[i],c[i]);
    }
    int ans = INT_MAX;
    for(int j=2;j<7;j++){
        if(j != 1 && j !=i && (1<<j)&mask ){
            ans = min(ans,fun(j, mask & ~(1<<j),r,c) + dis(r[i],c[i],r[j],c[j]));
        }
    }
    return ans;
}
int main(){
    vector<int>r{0,0,70,30,10,90,50,100};
    vector<int>c{0,0,40,10,5,70,20,100};
    int n = 8;
    int ans = INT_MAX;
    for(int i=2;i<7;i++){
        ans = min(ans,fun(i,(1<<7)-1,r,c)+dis(r[1],c[1],r[i],c[i]));
    }
    cout<<ans;
    return 0;
}