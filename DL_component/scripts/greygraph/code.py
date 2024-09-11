class Solution:
    def __init__(self):
        self.ans = []
        self.track = []

    def backtrack(self, cur, n, k):
        if len(self.track) == k:
            print(self.track)
            self.ans.append(self.track[:])
            return
        for i in range(cur, n + 1):
            self.track.append(i)
            self.backtrack(i + 1, n, k)
            self.track.pop()

    def combine(self, n: int, k: int) -> list[list[int]]:
        self.backtrack(1, n, k)
        return self.ans
    
S = Solution()
left = 2
right = 4
gas = [1,2,3,4,5]
next = [] * 26 
cost = [3,4,5,1,2]
s = "leetcode"
questions = [[1,1],[2,2],[3,3],[4,4],[5,5]]
S.combine(4, 2)