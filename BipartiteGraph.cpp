/*Bipartite Graph*/
#include<iostream>
#include<vector>
#include<climits>
#include<queue>
using namespace std;
void Bipartite(vector<vector<int>>&adjMatrix,int V){
    queue<int>q;
    q.push(0);
    vector<int>col(V,-1);
    col[0] = 0; 
    while(!q.empty()){
        auto u = q.front();
        q.pop();
        for(int i = 0 ;i<V;i++){
            if(adjMatrix[u][i]==1){
                if(col[i] == -1){
                    col[i] = !col[u];
                    q.push(i);
                }
                else{
                    if(col[u] == col[i]){
                        cout << -1;
                        return ;
                    }
                }
            }
        }
    }
    for(int i=0;i<V;i++){
        if(col[i]==0){
        cout << i;
        }
    }
    for(int i=0;i<V;i++){
        if(col[i]==1){
        cout << i;
        }
    }
}
int main(){
    int t;
    cin>>t;
    while(t-->0){
        int V;
        cin>>V;
        vector<vector<int>>adjMatrix(V+1,vector<int>(V+1,0));
        for(int i=0;i<V;i++){
            for(int j=0;j<V;j++){
                cin>>adjMatrix[i][j];
            }
        }
        Bipartite(adjMatrix,V);
    }
    return 0;
}