import itertools
a = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# a = "ABCDEFK"
b = list(itertools.permutations(a,3))
def check_score_all(a,b):
    p = 0
    for i in range(3):
        if a[i] == b[i]:
            p+=1
    # print("B", a, b, p)
    return p

def check_score_only_pos(a,b):
    p = 0
    for i in a:
        for j in b:
            if i == j:
                p+=1
    return p

def check(k, b, pos_only, score_req):
    c = []
    for i in b:
        if check_score_only_pos(k, i) == pos_only and check_score_all(k,i) == score_req:
            c.append(i)
    return c
c = []
problem_set = [
    #格式=(密码,所有可能性,号码正确量，位置正确量)
    ("ABC", b, 1, 1),
    ("AEF", b, 1, 0),
    ("CKA", b, 2, 0),
    ("DEB", b, 0, 0),
    ("BDK", b, 1, 0)
]

def check_all(ps):
    a = [i for i in b]
    for j in ps:
        a = list(set(a) & set(check(*j)))
    return a

check_all(problem_set)

