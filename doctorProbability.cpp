#include <iostream>
#include<vector>
#include<queue>
#include<map>
using namespace std;
int main() {
    int V;
    cin>>V;
    vector<vector<double>>adjMatrix(V+1,vector<double>(V+1,0));
    for(int i=0;i<V;i++){
        for(int j=0;j<V;j++){
            cin>>adjMatrix[i][j];
        }
    }
    int t ;
    cin >> t;
    //bfs
    queue<pair<int,double>>q;
    q.push({0,1.0});
    double maxi = 1e-9;
        while(!q.empty()){
            if(t<=0){
                
                while(!q.empty()){
                    maxi = max(maxi,q.front().second);
                    q.pop();
                }
                cout << maxi;
                break;
            }
            int n = q.size();
            map<int,double>mp;
            for(int i=0;i<n;i++){
                auto node = q.front();
                int u = node.first; 
                double p = node.second; 
                q.pop();
                for(int v=0;v<V;v++){
                    if(adjMatrix[u][v] != 0.0){
                        mp[v] += (p*adjMatrix[u][v]);
                    }
                }
                for(auto temp : mp ){
                    q.push({temp.first,temp.second});
                }
            }
            t-=10;
        }
        if(maxi == 1e-9){
            cout<<-1;
        }

    return 0;
}
