#include<iostream>
#include<vector>
#include<queue>
#include<climits>
using namespace std;
int finalAnswer = INT_MAX;
void resest(vector<int>&fishingSpots,int index){
    for(int i=0;i<fishingSpots.size();i++){
        if(fishingSpots[i] == index+1){
            fishingSpots[i] = 0;
        }
    }
}
int calculateDistance(int index,int &npos,int &pos1,int &pos2,vector<int>&fishingSpots,int loc,int fishers){
    int sum = 0;
    int left = loc ;
    int right = loc;
    int n = fishingSpots.size();
    int left_dst = INT_MAX;
    int right_dst = INT_MAX;
    for(int i=1;i<fishers;i++){
        if(fishingSpots[loc] == 0 ){
            fishingSpots[loc] = index + 1;
            sum++;
        }
        else{
            while(left > 0 && fishingSpots[left]>0){
                left--;
            }
            while(right <= n && fishingSpots[right]>0){
                right++;
            }
            if(left > 0 && fishingSpots[left] ==0){
            left_dst = loc - left + 1;
            }
            if(right <= n && fishingSpots[right] ==0){
            right_dst = right - loc +1;
            }
            if(left_dst == right_dst){
                if( i+1 < fishers){
                    fishingSpots[left] = index+1;
                    fishingSpots[right] = index+1;
                    sum += 2*left_dst;
                    i++;
                    if(i==fishers){
                        npos = 1;
                        return sum;
                    }
                }
                else{
                    fishingSpots[left] = index+1;
                    sum += left_dst;
                }
            }
            else if(left_dst < right_dst){
                fishingSpots[left]=index+1;
                sum += left_dst;
            }
            else{
                fishingSpots[right] = index+1;
                sum += right_dst;
            }
        }
    }
    left_dst = INT_MAX;
    right_dst = INT_MAX;
    while(left > 0 && fishingSpots[left]>0){
                left--;
            }
    while(right <= n && fishingSpots[right]>0){
                right++;
    }
    if(left > 0 && fishingSpots[left] ==0 ){
    left_dst = loc - left + 1;
    }
    if(right <=n && fishingSpots[right] == 0){
    right_dst = right - loc +1;
    }
    if(left_dst == right_dst){
        fishingSpots[left] = index+1;
        sum += left_dst;
        npos = 2;
        pos1 = left;
        pos2 = right;
    }
    else if(left_dst < right_dst){
        sum += left_dst;
        npos = 1;
        fishingSpots[left] = index+1;
    } 
    else{
        sum += right_dst;
        npos = 1 ;
        fishingSpots[right] = index+1;
    }
    return sum;
}
void solve(int index,int ans,vector<int>&fishingSpots,vector<int>&gates,vector<int>&fishersMans,vector<int>&vis,int count){
    
    int pos1;
    int pos2; 
    int npos;
    int sum = calculateDistance(index,npos,pos1,pos2,fishingSpots,gates[index],fishersMans[index]);
    
    if(count == 3){
        
        finalAnswer = min(ans+sum,finalAnswer);
        cout<<ans+sum<<"\n";
        for(int i=0;i<fishingSpots.size();i++){
            cout<<fishingSpots[i]<<" ";
        }
        return ;

    }

    if(npos == 1){
    for(int i=0;i<3;i++){
        if(vis[i]==0){ // add vis in function call from main
            vis[i] = 1;
            solve(i,ans+sum,fishingSpots,gates,fishersMans,vis,count+1);
            resest(fishingSpots,i);
            vis[i]=0;
        } 
    }
    }
    else{
         fishingSpots[pos1] = index+1;
         for(int i=0;i<3;i++){
                if(vis[i]==0){ // add vis in function call from main
                vis[i] = 1;
                solve(i,ans+sum,fishingSpots,gates,fishersMans,vis,count+1);
                resest(fishingSpots,i);
                vis[i]=0;
                } 
         }
        fishingSpots[pos1] = 0;
        fishingSpots[pos2] = index+1;
         for(int i=0;i<3;i++){
                if(vis[i]==0){ // add vis in function call from main
                vis[i] = 1;
                solve(i,ans+sum,fishingSpots,gates,fishersMans,vis,count+1);
                resest(fishingSpots,i);
                vis[i]=0;
                } 
            }
        fishingSpots[pos2] = 0; 
    }

}
int main(){
    int n;
    cin>>n;
    vector<int>fishingSpots(n+2,0);
    int m = 3;
    vector<int>gates(m+1,0);
    for(int i=0;i<m;i++){
        cin>>gates[i];
    }
    vector<int>fishersMans(m+1,0);
    for(int i=0;i<m;i++){
        cin>>fishersMans[i];
    }
    int ans = 0 ;
    vector<int>vis(m+1,0);
    for(int i=0;i<m;i++){
        vis[i] = 1;
        solve(i,ans,fishingSpots,gates,fishersMans,vis,1); 
        resest(fishingSpots,i);
        vis[i] = 0;
    }
    cout<<finalAnswer;
    return 0;
}