def lbp(img):
    w,h=img.size
    a=np.ones((w-1,h-1))
    m=np.array(img)
    for y in range(1,h-1):
        for x in range(1,w-1):
            vec=[m[y-1][x+1],m[y][x+1],m[y+1][x+1],m[y+1][x],m[y+1][x-1],m[y][x-1],m[y-1][x-1],m[y-1][x]]
            c=0
            for v in range(0,len(vec)):
                if(m[y][x]<=vec[v]):
                    c=c+2**v
            a[y-1][x-1]=c
    return a
def cambios(vec):
    c=0
    for i in range(0,len(vec)-1):
        if vec[i]!=vec[i+1]:
            c=c+1
    return c

def lbp_unif(img):
    w,h= img.size #tamaño imagen
    lbp=np.ones((w-1,h-1)) # Matri de 1s
    m=np.array(img) #imagen to array
    
    # Recorrer la matriz
    for y in range(1,h-1):
        for x in range(1,w-1):
            #Vector con pixeles vecinos
            vec=[m[y-1][x+1],m[y][x+1],m[y+1][x+1],m[y+1][x],m[y+1][x-1],m[y][x-1],m[y-1][x-1],m[y-1][x]]
            vec2=[m[y-1][x+1],m[y][x+1],m[y+1][x+1],m[y+1][x],m[y+1][x-1],m[y][x-1],m[y-1][x-1],m[y-1][x]]
            c=0 #valor del nuevo pixel
            for v in range(0,len(vec)):
                if(m[y][x]<=vec[v]):
                    vec2[v]=1
                    c=c+2**v
                else:
                    vec2[v]=0
            if(cambios(vec2)<=2):
                lbp[y-1][x-1]=c
            else:
                lbp[y-1][x-1]=256
    return lbp

def hist_lbp(image):
    img=lbp_unif(image).ravel()
    w,h=image.size
    key=sorted(list(collections.Counter(img).keys()))
    val=collections.Counter(img)
    hist=[]
    for i in range(0,len(key)):
        hist.append(int(val[key[i]]))
    hist=np.array(hist)/((w-1)*(h-1))
    key=np.array(key)
    return pd.DataFrame({'key':key,'val':hist})