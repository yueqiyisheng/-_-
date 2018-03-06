# coding='UTF-8'
def countAndSay(n):
    """
    :type n: int
    :rtype: str
    """
    i=1
    s='1'
    while(i<n):
        new = ''
        num=[]
        for j in range(len(s)):
            if s[j] in new:
                num[new.find(s[j])]+=1
            else:
                new = new+s[j]
                num.append(1)
        s=''
        for k in range(len(new)):
            s=s+str(num[k])+new[k]
        i+=1
    return s