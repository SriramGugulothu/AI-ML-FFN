#include<iostream>
#include<vector>
#include<queue>
#include<climits>
using namespace std;
int main(){
    int n;
    cin>>n;
    vector<vector<int>>grid(n+1,vector<int>(n+1,0));
    vector<vector<int>>dis(n+1,vector<int>(n+1,0));
    vector<vector<bool>>vis(n+1,vector<bool>(n+1,false));
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            cin>>grid[i][j];
        }
    }
    queue<pair<int,int>>q;
    q.push({0,0});
    vector<int>rows{0,0,1,-1};
    vector<int>cols{1,-1,0,0};
    vis[0][0] = true;
    while(!q.empty()){
        auto cell = q.front();
        int r = cell.first;
        int c = cell.second;
        q.pop();
        int nR,nC;
        for(int i=0;i<4;i++){
            nR = r+rows[i];
            nC = c+cols[i];
            if(nR>=0 && nR < n && nC >= 0 && nC <n && !vis[nR][nC] && grid[nR][nC] == 1){
                if(i == 0 || i==1){ //greedy thought
                    vis[nR][nC] = true;
                    dis[nR][nC] = dis[r][c];
                    q.push({nR,nC});
                }
                else{
                    vis[nR][nC] =true;
                    dis[nR][nC] = dis[r][c]+1;
                    q.push({nR,nC});
                }
            }
        }
    }
    cout<<dis[n-1][n-1];
    return 0;
}