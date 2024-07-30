#include <cmath>
#include <cstdio>
#include <vector>
#include <iostream>
#include <algorithm>
#include <climits>
using namespace std;
int ans = INT_MAX;
int dis(int r1,int c1,int r2,int c2){
    return abs(r1-r2)+abs(c2-c1);
}
void fun(int  sr,int sc,int tr,int tc,int value,vector<bool>&vis,vector<vector<int>>&warmHoles,int n){
    ans = min(ans,dis(sr,sc,tr,tc)+value);
    
    for(int i=0;i<n;i++){
        if(vis[i] == false )
        {
        vis[i] = true;
        fun(warmHoles[i][2],warmHoles[i][3],tr,tc,value+dis(sr,sc,warmHoles[i][0],warmHoles[i][1])+warmHoles[i][4],vis,warmHoles,n);
        
        fun(warmHoles[i][0],warmHoles[i][1],tr,tc,value+dis(sr,sc,warmHoles[i][2],warmHoles[i][3])+warmHoles[i][4],vis,warmHoles,n);
        vis[i] = false;
        }
    }
}

int main() {
    int t;
    cin>>t;
    while(t-->0){
        int n;
        cin>>n;
        int sr,sc,tr,tc;
        cin>>sr>>sc>>tr>>tc;
        vector<vector<int>>warmHoles(n+1,vector<int>(5));
        for(int i=0;i<n;i++){
            for(int j=0;j<5;j++){
            cin >> warmHoles[i][j];
            }
        }
        vector<bool>vis(n+1,false);
        fun(sr,sc,tr,tc,0,vis,warmHoles,n);
        cout<<ans<<"\n";
        ans = INT_MAX;
    }
    return 0;
}
