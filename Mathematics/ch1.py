# chapter 1


# the tower of Hanoi

def towerHanoi(n):
    if n==0:
        return 0
    return 2*towerHanoi_1(n-1)+1

def towerHanoiEq(n):
    return 2**n-1

# lines in the plane
def linesPlane(n):
    if n==0:
        return 1;
    return linesPlane(n-1)+n
def linesPlaneEq(n):
    return n*(n+1)/2+1

# the Josephus problem

def josephus(n):
    # n>=1
    if n==1:
        return 1
    if n%2==0:
        return 2*josephus(n/2)-1
    return 2*josephus((n-1)/2)+1

#使用切片，简单易理解，速度最快，可同时实现循环左移（k>0）和右移（k<0）。
def conversion(lst,k):
    return lst[k:]+lst[:k]
def josephusConversion(n):
    n_str = decimalToAny(n,2) #bin(n)
    n_str = conversion(n_str,1)
    return int(n_str,2)


def josephusExtension(n, alpha,beta):
    # beta is a list
    n_str = decimalToAny(n,2)
    new_sum = 0
    new_factor = 1
    for i in range(len(n_str)-1,0,-1):
        #new_str = str(alpha) + str(beta[int(n_str[i])])
        new_sum = new_sum + new_factor * beta[int(n_str[i])]
        new_factor = new_factor*2
    new_sum = new_sum + new_factor * alpha
    return new_sum #anyToDecimal(new_str,2)


def josephusGeneral(n, d,c, alpha,beta):
    print "n = "+str(n)
    # conversion: n to base-d
    baseStr = {10:'a', 11:'b', 12:'c', 13:'d', 14:'e',
               15:'f', 16:'g', 17:'h', 18:'i', 19:'j',
               20:'k', 21:'l', 22:'m', 23:'n', 24:'o',
               25:'p', 26:'q', 27:'r', 28:'s', 29:'t',
               30:'u', 31:'v', 32:'w', 33:'x', 34:'y', 35:'z'}
    base_d_str = '';
    while n != 0:
        remainder = n%d
        if 0<=remainder<=9:
            remainder_string = str(remainder)
        elif 9<remainder<36:
            remainder_string = baseStr[remainder]
        else: #remainder>=36
            remainder_string = '('+str(remainder)+')'
        base_d_str = remainder_string + base_d_str
        n = n/d
    print "base-"+str(d)+": " + base_d_str
    # conversion to base-c
    base_c_num = 0
    factor = 1
    for i in range(len(base_d_str)-1,0,-1):
        base_c_num = base_c_num + factor * beta[int(base_d_str[i])]
        factor = factor * c
    base_c_num = base_c_num + factor * alpha[int(base_d_str[0])-1]
    print "f(n) = " + str(base_c_num)
    return base_c_num


def demo(): #e.g.
    alpha = [34,5]
    beta = [76,-2,8]
    print josephusGeneral(19,3,10,alpha,beta)


def decimalToAny(num,n):
    baseStr = {10:"a", 11:"b", 12:"c", 13:"d", 14:"e",
               15:"f", 16:"g", 17:"h", 18:"i", 19:"j"}
    new_num_str = ""
    while num != 0:
        remainder = num%n
        if 20 > remainder > 9:
            remainder_string = baseStr[remainder]
        elif remainder >= 20:
            remainder_string = "("+str(remainder)+")"
        else:
            remainder_string = str(remainder)
        new_num_str = remainder_string + new_num_str
        num = num/n
    return new_num_str
#将任意进制数转换成十进制
def anyToDecimal(num,n):
    # num = "", char type
    baseStr = {"0":0, "1":1, "2":2, "3":3, "4":4,
               "5":5, "6":6, "7":7, "8":8, "9":9,
               "a":10, "b":11, "c":12, "d":13, "e":14,
               "f":15, "g":16, "h":17, "i":18, "j":19}
    new_num = 0
    nNum = len(num)-1
    for i in num:
        new_num = new_num + baseStr[i]*pow(n,nNum)
        nNum = nNum -1
    return new_num
