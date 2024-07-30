//{ Driver Code Starts
#include<bits/stdc++.h>
using namespace std;

// } Driver Code Ends
class Solution 
{
    public:
    //Function to find out minimum steps Knight needs to reach target position.
	int minStepToReachTarget(vector<int>&KnightPos,vector<int>&TargetPos,int N)
	{
	    // Code here
	    vector<int>rows = {-2,-1,1,2,2,1,-1,-2};
	    vector<int>cols = {1,2,2,1,-1,-2,-2,-1};
	    queue<vector<int>>q;
	    vector<vector<bool>>vis(N+1,vector<bool>(N+1,false));
	    q.push({KnightPos[0],KnightPos[1],0});
	    
	    if(KnightPos[0]==TargetPos[0] && KnightPos[1]==TargetPos[1]){
	        return 0;
	    }
	    vis[KnightPos[0]][KnightPos[0]] = true; 
	    while(!q.empty()){
	        auto ele = q.front();
	        q.pop();
	        for(int i=0;i<8;i++){
	            int nRow = ele[0] + rows[i];
	            int nCol = ele[1] + cols[i];
	            int steps = ele[2];
	            if(nRow<=N && nRow>=0 && nCol>=0 && nCol<=N){
	                if(nRow == TargetPos[0] && nCol == TargetPos[1]){
	                    return steps+1;
	                }
	                if(vis[nRow][nCol]==false){
	                q.push({nRow,nCol,steps+1});
	                vis[nRow][nCol] = true;
	                }
	            }
	        }
	    }
	    return -1;
	}
};

//{ Driver Code Starts.
int main(){
	int tc;
	cin >> tc;
	while(tc--){
		vector<int>KnightPos(2);
		vector<int>TargetPos(2);
		int N;
		cin >> N;
		cin >> KnightPos[0] >> KnightPos[1];
		cin >> TargetPos[0] >> TargetPos[1];
		Solution obj;
		int ans = obj.minStepToReachTarget(KnightPos, TargetPos, N);
		cout << ans <<"\n";
	}
	return 0;
}
// } Driver Code Ends