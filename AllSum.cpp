#include<iostream>
#include<vector>
#include<queue>
#include<climits>
using namespace std;
int finalAnswer = 0;
bool isSafe(int r,int c, int n){
    if(r<0 || r>=n || c<0 || c>=n){
        return false;
    }
    return true;
}
void fun(int r,int c,int n,vector<vector<int>> &maze,vector<vector<bool>> &vis,vector<vector<int>> &path,int ans){
    
    if(r == n-1 && c == n-1){
        if(ans > finalAnswer){
        for(int i=0;i<n;i++){
            for(int j=0;j<n;j++){
                if(vis[i][j]){
                    path[i][j] = 3;
                }
            }
        }
        answer = finalAnswer;
        }
        return;
    }
    if(r<0 || r>=n || c<0 || c>=n){
        return ;
    }
    vector<int>rows{-1,1,0,0};
    vector<int>cols{0,0,-1,1};
    int nR,nC;
    vis[r][c] = true;
    for(int i=0;i<4;i++){
        nR = rows[i] + r;
        nC = cols[i] + c;
        if(isSafe(nR,nC,n) && vis[nR][nC] == false && maze[nR][nC] != 1){
            fun(nR,nC,n,maze,vis,path,ans+maze[nR][nC]);
        }
    }    
    vis[r][c] = false;    
}
int main(){
    int n;
    cin>>n;
    vector<vector<int>>maze(n+1,vector<int>(n+1,0));
    vector<vector<bool>>vis(n+1,vector<bool>(n+1,0));
    vector<vector<int>>path(n+1,vector<int>(n+1,0));
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            cin >> maze[i][j];
        }
    }
    int ans = 0 ;
    fun(0,0,n,maze,vis,path,ans);
    cout << finalAnswer <<"\n";
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            cout<<path[i][j]<<" ";
        }
        cout<<"\n";
    }
    return 0;
}