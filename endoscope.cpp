#include<iostream>
#include<vector>
#include<queue>
using namespace std;
struct a {
	bool left = false;
	bool right = false;
	bool up = false;
	bool down = false;
};
bool isSafe(int u,int v,int m,int n){
	if(u<0 || u>=m || v<0 || v>=n){
		return false;
	}
	return true;
}
int main(){
	int t;
	cin>>t;
	while(t-- > 0){
		int m,n;
		cin>>m>>n;
		int r,c;
		cin>>r>>c;
		int steps;
		cin>>steps;
		vector<vector<int>>arr(m,vector<int>(n));
		struct a policy[m+1][n+1];
		for(int i=0;i<m;i++){
			for(int j=0;j<n;j++){
				cin>>arr[i][j];
				if(arr[i][j]==1){
					policy[i][j].left = true;
					policy[i][j].right = true;
					policy[i][j].up = true;
					policy[i][j].down = true;
				}
				else if(arr[i][j]==2){
					policy[i][j].up = true;
					policy[i][j].down = true;
				}
				else if(arr[i][j]==3){
					policy[i][j].left = true;
					policy[i][j].right = true;
				}
				else if(arr[i][j]==4){
					policy[i][j].up = true;
					policy[i][j].right = true;
				}
				else if(arr[i][j]==5){
					policy[i][j].right = true;
					policy[i][j].down = true;
				}
				else if(arr[i][j]==6){
					policy[i][j].left = true;
					policy[i][j].down = true;
				}
				else if(arr[i][j]==7){
					policy[i][j].up = true;
					policy[i][j].left = true;
				}
			}
		}
		
		vector<vector<bool>>vis(m,vector<bool>(n,false));
		vis[r][c] = true;
		queue<pair<int,int>>q;

		q.push({r,c});
		
		int ans = 1;
		steps--;
		if(steps == 0){
			cout<<0<<"\n";
			continue;
		}
		if(arr[r][c]==0){
			cout<<0<<"\n";
			continue;
		}
		while(!q.empty()){
			int nU = 0;
			int nV = 0;
			if(steps == 0 ){
				break;
			}
			int n2 = q.size();
			for(int i=0;i<n2;i++){
				auto node = q.front();
				q.pop();
				int u = node.first;
				int v = node.second;
				if(arr[u][v]==1){
					nU = u-1;
					nV = v;
					if(isSafe(nU,nV,m,n) && policy[nU][nV].down && !vis[nU][nV]){
						q.push({nU,nV});
						vis[nU][nV] = true;
						ans++;
					}
					nU = u+1;
					nV = v;
					if(isSafe(nU,nV,m,n) && policy[nU][nV].up && !vis[nU][nV]){
						q.push({nU,nV});
						vis[nU][nV] = true;
						ans++;
						
					}
					nU = u;
					nV = v-1;
					if(isSafe(nU,nV,m,n) && policy[nU][nV].right && !vis[nU][nV]){
						q.push({nU,nV});
						vis[nU][nV] = true;
						ans++;
					}
					nU = u;
					nV = v+1;
					if(isSafe(nU,nV,m,n) && policy[nU][nV].left && !vis[nU][nV]){
						q.push({nU,nV}); 
						vis[nU][nV] = true;
						ans++;
					}
				}

				else if(arr[u][v]==2){
					nU = u-1;
					nV = v;
					if(isSafe(nU,nV,m,n) && policy[nU][nV].down && !vis[nU][nV]){
						q.push({nU,nV});
						vis[nU][nV] = true;
						ans++;
					}
					nU = u+1;
					nV = v;
					if(isSafe(nU,nV,m,n) && policy[nU][nV].up && !vis[nU][nV]){
						q.push({nU,nV});
						ans++;
						vis[nU][nV]= true;
					}
				}

				else if(arr[u][v]==3){
					nU = u;
					nV = v-1;
					if(isSafe(nU,nV,m,n) && policy[nU][nV].right && !vis[nU][nV]){
						q.push({nU,nV});
						vis[nU][nV] = true;
						ans++;
					}
					nU = u;
					nV = v+1;
					if(isSafe(nU,nV,m,n) && policy[nU][nV].left && !vis[nU][nV]){
						q.push({nU,nV});
						vis[nU][nV] = true;
						ans++;
					}
				}

				else if(arr[u][v]==4){
					nU = u-1;
					nV = v;
					if(isSafe(nU,nV,m,n) && policy[nU][nV].down && !vis[nU][nV]){
						q.push({nU,nV});
						vis[nU][nV] = true;
						ans++;
					}
					nU = u;
					nV = v+1;
					if(isSafe(nU,nV,m,n) && policy[nU][nV].left && !vis[nU][nV]){
						q.push({nU,nV});
						vis[nU][nV] = true;
						ans++;
					}
				}

				else if(arr[u][v]==5){
					nU = u;
					nV = v+1;
					if(isSafe(nU,nV,m,n) && policy[nU][nV].left && !vis[nU][nV]){
						q.push({nU,nV});
						vis[nU][nV] = true;
						ans++;
					}
					nU = u+1;
					nV = v;
					if(isSafe(nU,nV,m,n) && policy[nU][nV].up && !vis[nU][nV]){
						q.push({nU,nV});
						vis[nU][nV] = true;
						ans++;
					}
				}

				else if(arr[u][v]==6){
					nU = u;
					nV = v-1;
					if(isSafe(nU,nV,m,n) && policy[nU][nV].right && !vis[nU][nV]){
						q.push({nU,nV});
						vis[nU][nV] = true;
						ans++;
					}
					nU = u+1;
					nV = v;
					if(isSafe(nU,nV,m,n) && policy[nU][nV].up && !vis[nU][nV]){
						q.push({nU,nV});
						vis[nU][nV] = true;
						ans++;
						
					}
				}

				else if(arr[u][v]==7){
					nU = u;
					nV = v-1;
					if(isSafe(nU,nV,m,n) && policy[nU][nV].right && !vis[nU][nV]){
						q.push({nU,nV});
						vis[nU][nV] = true;
						ans++;
					}
					nU = u-1;
					nV = v;
					if(isSafe(nU,nV,m,n) && policy[nU][nV].down && !vis[nU][nV]){
						q.push({nU,nV});
						vis[nU][nV] = true;
						ans++;
					} 
				}
			}
		  steps--;
		}
		cout<<ans<<"\n";	
	}
	return 0;
}