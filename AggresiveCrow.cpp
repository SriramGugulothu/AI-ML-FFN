/*aggresive crow*/
#include<iostream>
#include<vector>
#include<climits>
using namespace std;
bool check(int value, vector<int>&pos,int c){
    int ele = pos[0];
    int count = 1;
    bool flag = false;
   for(int i=1;i<pos.size();i++){
        if(pos[i]-ele >= value){
            ele = pos[i];
            count++;
        }
        
    }

    if(count >= c){
        return true;
    }
    return false;
}
int main(){
    int t;
    cin>>t;
    while(t-->0){
        int n;
        cin>>n;
        vector<int>pos(n+1,0);
        int maxi = INT_MIN;
        int mini = INT_MAX;
        for(int i=0;i<n;i++){
            cin>>pos[i];
            maxi = max(maxi,pos[i]);
            mini = min(mini,pos[i]);
        }
        int low = 0;
        int high = maxi - mini;
        int c;
        cin>>c;
        int ans = -1;
        while(low<=high){
            int mid = (low+high)/2;
            if(check(mid,pos,c)==true){
                ans = mid;
                low = mid+1;
            }
            else{
                high = mid-1;
            }
        }
        cout<<ans;
    }
    return 0;
}