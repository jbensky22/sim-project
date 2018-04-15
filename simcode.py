
    
def pred_win_diff():
    import pickle
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn import datasets, linear_model
    from sklearn.metrics import mean_squared_error, r2_score

    wins =pickle.load( open( "wins1shortor.p", "rb" ) )
    wins2=pickle.load( open( "wins2shortor.p", "rb" ) )

    runs =pickle.load( open( "runs1shortor.p", "rb" ) )
    runs=pickle.load( open( "runs2shortor.p", "rb" ) )

    ip1 =pickle.load( open( "ip1shortor.p", "rb" ) )
    era1 =pickle.load( open( "era1shoror.p", "rb" ) )
    ip2 =pickle.load( open( "ip2shortor.p", "rb" ) )
    era2 =pickle.load( open( "era2shortor.p", "rb" ) )
    runs1=pickle.load( open( "runs1shortor.p", "rb" ) )
    runs2=pickle.load( open( "runs2shortor.p", "rb" ) )

    windiff= [a - b for a, b in zip(wins2, wins)]

    regr = linear_model.LinearRegression()

    spavg=[]
    spstd=[]
    rpavg=[]
    rpstd=[]
    z=[]
    o=[]
    t=[]
    th=[]
    f=[]
    te=[]
    el=[]
    
    d= pickle.load( open( "backgroundshortorder.p", "rb" ) )

    for i in range(0,30):
        sp=[]
        rp=[]
        for p in range(0,12):
            if p<5:
                sp.append(d[i][1][p][0])
            else:
                rp.append(d[i][1][p][0])
        z.append(d[i][1][0][0])
        o.append(d[i][1][1][0])
        t.append(d[i][1][2][0])
        th.append(d[i][1][3][0])
        f.append(d[i][1][4][0])
        te.append(d[i][1][10][0])
        el.append(d[i][1][11][0])
        spstd.append(np.std(sp))
        rpstd.append(np.std(rp))
        spavg.append(np.mean(sp))
        rpavg.append(np.mean(rp))

    dtype = [('wins','float32'), ('spstd','float32'), ('spavg','float32'),('rpstd','float32'), ('rpavg','float32')]
    
        
##    wins=np.reshape(wins,(-1,1))
##    spstd=np.reshape(spstd,(-1,1))
##    spavg=np.reshape(spavg,(-1,1))
##    rpstd=np.reshape(rpstd,(-1,1))
##    rpavg=np.reshape(rpavg,(-1,1))
##    a=[wins,spstd,spavg,rpstd,rpavg]
##    pred=np.array(a)
    rundiff= [a - b for a, b in zip(runs1, runs2)]
    df= {'rundiff':rundiff,'windiff': windiff,'wins': wins, 'spstd': spstd, 'spavg': spavg, 'rpstd': rpstd, 'rpavg': rpavg,'ace':z,'ntwo':o,'nthree':t,'nfour':th,'nfive':f,'onemop':te,'twomop':el}
 

    dfout= {'windiff': windiff}
    df = pd.DataFrame(data=df)
    
    df.to_csv('simdata.csv')

    dist=[]
    for i in range(0,30):
        pitchers=[z[i],o[i],t[i],th[i],f[i],te[i],el[i]]
        dist.append(pitchers)


    from matplotlib.pyplot import subplots, show
    windiff=sorted(zip(windiff,range(0,30)), reverse=False)
    
##
    fig, ax = subplots()
    low=[0,0,0,0,0,0,0]
    high=[0,0,0,0,0,0,0]
    for i in range(0,30):
        if i<10:
            for t in range(0,7):
                low[t]+=(dist[windiff[i][1]][t])
        if i>10 and i<21:
            for t in range(0,7):
                high[t]+=(dist[windiff[i][1]][t])
    for t in range(0,7):
        low[t]=float(float(low[t])/float(10))
        high[t]=float(float(high[t])/float(10))
    ax.plot(high,'b-')
    ax.plot(low,'r-')
##
    show()
        
    return

    
    nsamples, nx, ny = pred.shape
    pred = pred.reshape((nx*ny,nsamples))
    windiff=np.reshape(windiff,(-1,1))
    print(np.shape(windiff))
    print(np.shape(pred))
    

    regr.fit(windiff, pred)
    regr.fit([[getattr(t, 'x%d' % i) for i in range(1, 6)] for t in pred],
        [t.y for t in pred])

    

    windiff_pred = regr.predict(pred)
    print(windiff_pred)

# The coefficients
    print('Coefficients: \n', regr.coef_)
# The mean squared error
    print("Mean squared error: %.2f"
      % mean_squared_error(windiff, windiff_pred))
# Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(windiff, windiff_pred))


### test the simulation once
def main2():
    import pickle
    d = pickle.load( open( "backgroundshortorder.p", "rb" ) )

    
    
    return(sim(d,1))
## run the whole simulation, isolating one team at a time to use the new strategy, while all other teams use the old strategy 
def main():
    
    import pickle
    from scipy.stats.stats import pearsonr

    d=background()
    pickle.dump( d, open( "backgroundshortorder.p", "wb" ) )
   

    wins=[]
    wins2=[]
    ip1=[]
    era1=[]
    ip2=[]
    era2=[]
    runs1=[]
    runs2=[]
    acediff=[]
    for team in range(0,30):
        print(team)
        
        
            
        d = pickle.load( open( "backgroundshortorder.p", "rb" ) )
        for i in range(0,100):
            d=sim(d,team)
        print('ace run diff')
        adiff=float(float(d[team][3][0][2])/float(d[team][3][0][1]))-float(float(d[team][7][0][2])/float(d[team][7][0][1]))
        print(adiff)
        acediff.append(adiff)
        print('-----------------------')
        wins.append(float(float(d[team][4][0])/float(100)))
        wins2.append(float(float(d[team][8][0])/float(100)))
        r1=0
        r2=0
        for p in range(0,12):
            ip1.append(d[team][3][p][1])
            era1.append(d[team][1][p][0])
            
            ip2.append(d[team][7][p][1])
            era2.append(d[team][5][p][0])
               
    
            r1+=d[0][3][p][2]
            r2+=d[0][7][p][2]
        runs1.append(float(float(r1)/float(100)))
        runs2.append(float(float(r2)/float(100)))
        print(wins[team])
        print(wins2[team])
        
 
    ##.28
    print(pearsonr(ip1, era1))
    ##.39
    print(pearsonr(ip2, era2))
    
    
    pickle.dump(wins, open( "wins1shortor.p", "wb" ) )
    pickle.dump(wins2, open( "wins2shortor.p", "wb" ) )

    pickle.dump(acediff, open( "acediffshortor.p", "wb" ) )

    pickle.dump(ip1, open( "ip1shortor.p", "wb" ) )
    pickle.dump(era1, open( "era1shoror.p", "wb" ) )

    pickle.dump(ip2, open( "ip2shortor.p", "wb" ) )
    pickle.dump(era2, open( "era2shortor.p", "wb" ) )

    pickle.dump(runs1, open( "runs1shortor.p", "wb" ) )
    pickle.dump(runs2, open( "runs2shortor.p", "wb" ) )

## analyze results of the sim
def analyze():
    import pickle
    from scipy.stats.stats import pearsonr
    
    import matplotlib.pyplot as plt; plt.rcdefaults()
    import numpy as np
    import matplotlib.pyplot as plt
 
    teams = ('0', '1', '2', '3', '4', '5','6', '7', '8', '9', '10', '11','12', '13', '14', '15', '16', '17','18', '19', '20', '21', '22', '23','24', '25', '26', '27', '28', '29')
    wins =pickle.load( open( "wins1shortor.p", "rb" ) )
    wins2=pickle.load( open( "wins2shortor.p", "rb" ) )

    ip1 =pickle.load( open( "ip1shortor.p", "rb" ) )
    era1 =pickle.load( open( "era1shoror.p", "rb" ) )
    ip2 =pickle.load( open( "ip2shortor.p", "rb" ) )
    era2 =pickle.load( open( "era2shortor.p", "rb" ) )
    runs1 =pickle.load( open( "runs1shortor.p", "rb" ) )
    runs2 =pickle.load( open( "runs2shortor.p", "rb" ) )

    ###.39
    windiff= [a - b for a, b in zip(wins2, wins)]
    rundiff= [a - b for a, b in zip(runs2, runs1)]
    print(windiff)
    print(sum(windiff)/len(windiff))

 
    
## create the finctional league. draw randomly from sp and rp era distrbutions to create 12 pitchers on each team   
def background():
    import pandas as pd
    import numpy

    team_dict={}
    for i in range(0,30):
        team_dict[i]=[]

    spruns=pd.read_csv('spruns.csv')
    spruns=spruns['ER']/spruns['IP']
    rpruns=pd.read_csv('rpruns.csv')
    rpruns=rpruns['ER']/rpruns['IP']

    from collections import deque
 
    ## randomly assign teams run levels from this data
    teamprun=[]
    for i in range(0,30):
        prunl=[]
        spqueue=deque()
        rel=[]
        p_dict={}
        li_dict={}
        srel=[]
        
        for p in range(0,5):
            prun=numpy.random.normal(numpy.mean(spruns),numpy.std(spruns))
 #           prunl.append(prun)
            srel.append(prun)
            
            spqueue.append(p)
            li_dict[p]=[0,0,0,0]
            #0
        srel.sort()
        for s in range(0,5):
            p_dict[s]=srel[s]
        team_dict[i].append(spqueue)
        fi=srel[4]
        fo=srel[3]
        

        for p in range(5,12):
            prun=numpy.random.normal(numpy.mean(rpruns),numpy.std(rpruns))
            prunl.append(prun)
            
            rel.append(prun)
            li_dict[p]=[0,0,0,0]
        rel.sort()
        for r in range(0,5):
            p_dict[r]=[p_dict[r],(deque())]
            for ii in range(0,7):
                p_dict[r][1].append(0)
        
        ## relivers is a dict. the value of each entry is a list. first entry of list is run talent. second entry is queue ip pitched over past 5 games
        for r in range(5,12):
            p_dict[r]=[rel[r-5]]
            p_dict[r].append(deque())
            for ii in range(0,7):
                p_dict[r][1].append(0)
        if p_dict[10][0]<p_dict[3][0]:
            holder=p_dict[10][0]
            holder2=p_dict[3][0]
            holder3=p_dict[4][0]
            p_dict[4][0]=holder2
            p_dict[3][0]=holder
            p_dict[10][0]=holder3
            if p_dict[11][0]<p_dict[4][0]:
                holder=p_dict[11][0]
                p_dict[11][0]=p_dict[4][0]
                p_dict[4][0]=holder
            
        elif p_dict[10][0]<p_dict[4][0]:
            
##            print(p_dict[10][0])
##            print(p_dict[4][0])
            holder=p_dict[10][0]
            p_dict[10][0]=p_dict[4][0]
            p_dict[4][0]=holder

        srel=[]
        
        for p in range(0,5):
            srel.append(p_dict[p][0])
        srel.sort()
        for s in range(0,5):
            p_dict[s][0]=srel[s]
            
        #1
        team_dict[i].append(p_dict)
        ## home games 2
        
        team_dict[i].append(0)
        ## li dict 3
        team_dict[i].append(li_dict)
        ## record 4
        team_dict[i].append([0,0])



    teamprun2=[]
    for i in range(0,30):
        prunl2=[]
        rel2=[]
        p_dict2={}
        li_dict2={}
        spqueue=deque()
        spqueue.append(1)
        spqueue.append(2)
        spqueue.append(3)
        spqueue.append(4)
        
        
        for p in range(0,12):
 #           prunl.append(prun)
            p_dict2[p]=[team_dict[i][1][p][0],deque()]
            for ii in range(0,7):
                p_dict2[p][1].append(0)
            li_dict2[p]=[0,0,0,0]
            #0
        

        #1
        team_dict[i].append(p_dict2)
        ## home games 2
        
        team_dict[i].append(0)
        ## li dict 3
        team_dict[i].append(li_dict2)
        ## record 4
        team_dict[i].append([0,0])
        team_dict[i].append(spqueue)
 
##        
    return(team_dict)



 ##### glossary
 ## team_dict has 4 enetries
 ### first entry is a queue of sp numbers
 ### second is dict of mappings from sp number to run talent for sp
 ### third is dict of reliever numbers to run talent and ip over past 3 games
 ### fourth is count of home games for team


 #   sim(team_dict)



def sim(team_dict,target):
    from collections import deque
 
    for t in range(0,1):
        
    
        for i in range(0,162):
            
            import numpy
            
            games=randmatchup()

            for g in games:

                if team_dict[g[0]][2]>team_dict[g[1]][2]:
                    away=g[0]
                    home=g[1]
                    team_dict[home][2]+=1
                else:
                    away=g[1]
                    home=g[0]
                    team_dict[home][2]+=1
                ## pick sp
                home_sp=team_dict[home][0].popleft()
                away_sp=team_dict[away][0].popleft()
                home_p=home_sp
                away_p=away_sp
                spinh=1
                spina=1

                home_runs=0
                away_runs_last_sub=0
                home_inn_last_sub=0

                away_runs=0
                home_runs_last_sub=0
        
                away_inn_last_sub=0
                ##home
                rphome=[10,11]
                homeremove=[]
                                    
                
                for p in homeremove:
                    if len(team_dict[home][1][p][1])==7:
                        team_dict[home][1][p][1].popleft()
                        team_dict[home][1][p][1].append(0)
                    else:
                        team_dict[home][1][p][1].append(0)
                        
                rpaway=[10,11]
                awayremove=[]
                

                
                for p in awayremove:
                    if len(team_dict[away][1][p][1])==7:
                        team_dict[away][1][p][1].popleft()
                        team_dict[away][1][p][1].append(0)
                    else:
                        team_dict[away][1][p][1].append(0)
                
                
 
                spin=1
                for inn in range(0,6):
                    ## home pitching
                    ### calculate li index before half inning starts
                    home_defc=home_runs-away_runs

                    ### picking relivers
                    if inn<=5:
                        
                        new_p=pull_pitcher(team_dict,home,spinh,home_p,inn,away_runs,home_runs,rphome,away_runs_last_sub,home_inn_last_sub,0)
                        #if new_p is None:
                            
                        if (spinh==1) and (new_p!=home_sp):
                            spinh=0
                            away_runs_last_sub=away_runs
                            home_last_inn_sub=inn
                            home_p=new_p
 #                           findli(inn,home_defc,1,home,away,home_p,away_p,team_dict,0)
                            team_dict[home][3][home_p][3]+=1
                            rphome.remove(new_p)
                            
     
                        elif (spinh==0) and (new_p!=home_p):    
                            if len(team_dict[home][1][home_p][1])==7:
                                team_dict[home][1][home_p][1].popleft()
                                team_dict[home][1][home_p][1].append(inn-home_last_inn_sub)
                            else:
                                team_dict[home][1][home_p][1].append(inn-home_last_inn_sub)
                            away_runs_last_sub=away_runs
                            home_last_inn_sub=inn
                            home_p=new_p
                            
                            rphome.remove(new_p)
                            
                           # findli(inn,home_defc,1,home,away,home_p,away_p,team_dict,0)
                            team_dict[home][3][home_p][3]+=1


                    findli(inn,home_defc,1,home,away,home_p,away_p,team_dict,0)
                    runprob=team_dict[home][1][home_p][0]
                    #runs=numpy.random.binomial(3,float(runprob)/float(3))
                    runs=numpy.random.poisson(float(runprob),1)[0]
                    away_runs+=runs
                    

                    team_dict[home][3][home_p][1]+=1
                    team_dict[home][3][home_p][2]+=runs

                    if inn==5:
                        if home_p!=home_sp:
                            if len(team_dict[home][1][home_p][1])==7:
                                team_dict[home][1][home_p][1].popleft()
                                team_dict[home][1][home_p][1].append(6-home_last_inn_sub)
                            else:
                                team_dict[home][1][home_p][1].append(6-home_last_inn_sub)
                       

                            
                    if inn==5 and home_runs>away_runs:
                        if away_p!=away_sp:
                            if len(team_dict[away][1][away_p][1])==7:
                                team_dict[away][1][away_p][1].popleft()
                                team_dict[away][1][away_p][1].append(5-away_last_inn_sub)
                            else:
                                team_dict[away][1][away_p][1].append(5-away_last_inn_sub)
                        

                        break


                    ## away pitching

                    away_defc=home_runs-away_runs

                    ### picking relivers
                    if inn<=5:
                        new_p=pull_pitcher(team_dict,away,spina,away_p,inn,home_runs,away_runs,rpaway,home_runs_last_sub,away_inn_last_sub,0)
                        if (spina==1) and (new_p!=away_sp):
                            spina=0
                            home_runs_last_sub=home_runs
                            away_last_inn_sub=inn
                            away_p=new_p
 #                           findli(inn,away_defc,0,home,away,home_p,away_p,team_dict,0)
                            team_dict[away][3][away_p][3]+=1
     
                            rpaway.remove(new_p)
                            
                        elif (spina==0) and (new_p!=away_p):
                            

                            if len(team_dict[away][1][away_p][1])==7:
                                team_dict[away][1][away_p][1].popleft()
                                team_dict[away][1][away_p][1].append(inn-away_last_inn_sub)
                            else:
                                team_dict[away][1][away_p][1].append(inn-away_last_inn_sub)
                            home_runs_last_sub=home_runs
                            away_last_inn_sub=inn
                            rpaway.remove(new_p)
                           
                            away_p=new_p
                            #findli(inn,away_defc,0,home,away,home_p,away_p,team_dict,0)
                            team_dict[away][3][away_p][3]+=1
     

                    findli(inn,away_defc,0,home,away,home_p,away_p,team_dict,0)                                       
                    runprob=team_dict[away][1][away_p][0]
#                       runs=numpy.random.binomial(3,float(runprob)/float(3))
                    runs=numpy.random.poisson(float(runprob),1)[0]
   
                    home_runs+=runs

                    team_dict[away][3][away_p][1]+=1
                    team_dict[away][3][away_p][2]+=runs

                    if inn==5:
                        if away_p!=away_sp:
                            if len(team_dict[away][1][away_p][1])==7:
                                team_dict[away][1][away_p][1].popleft()
                                team_dict[away][1][away_p][1].append(6-away_last_inn_sub)
                            else:
                                team_dict[away][1][away_p][1].append(6-away_last_inn_sub)
                            
                #if i<0:
#                    print(rphome)
##                    print('game')
##                    print('--------')
##                    print('home')
##                    print('--------')
##                    print(home_runs)
##                    print('--------')
##                    print(away_runs)
##                    print('--------')
##                    print('end game')
                team_dict[home][0].append(home_sp)
                team_dict[away][0].append(away_sp)
                if home_runs>away_runs:
                    team_dict[home][4][0]+=1
                    team_dict[away][4][1]+=1
                else:
                    team_dict[away][4][0]+=1
                    team_dict[home][4][1]+=1
                for p in rphome:
                    if len(team_dict[home][1][p][1])==7:
                        team_dict[home][1][p][1].popleft()
                        team_dict[home][1][p][1].append(0)
                    else:
                        team_dict[home][1][p][1].append(0)
               
                for p in rpaway:
                    if len(team_dict[away][1][p][1])==7:
                        team_dict[away][1][p][1].popleft()
                        team_dict[away][1][p][1].append(0)
                    else:
                        team_dict[away][1][p][1].append(0)
     
               
                

    for t in range(0,1):
        for i in range(0,162):
            import numpy
            games=randmatchup()

            for g in games:

                if team_dict[g[0]][6]>team_dict[g[1]][6]:
                    away=g[0]
                    home=g[1]
                    team_dict[home][6]+=1
                else:
                    away=g[1]
                    home=g[0]
                    team_dict[home][6]+=1
                if home==target:
                    sim3(team_dict,home,away,1,0)
                elif away==target:
                    sim3(team_dict,home,away,0,1)
                else:
                    sim3(team_dict,home,away,0,0)
    return(team_dict)
                
                
def sim3(team_dict,home,away,h,a):
    import itertools
    import numpy
    spinh=1
    spina=1

    home_runs=0
    away_runs_last_sub=0
    home_last_inn_sub=0

    away_runs=0
    home_runs_last_sub=0

    away_last_inn_sub=0
    if h==0 and a==0:
        home_sp=team_dict[home][0].popleft()
        away_sp=team_dict[away][0].popleft()
        home_p=home_sp
        away_p=away_sp
 
        ##home
        rphome=[10,11]
        homeremove=[]
                            
        
        for p in homeremove:
            if len(team_dict[home][5][p][1])==7:
                team_dict[home][5][p][1].popleft()
                team_dict[home][5][p][1].append(0)
            else:
                team_dict[home][5][p][1].append(0)
                
        rpaway=[10,11]
        awayremove=[]
        
        
        for p in awayremove:
            if len(team_dict[away][5][p][1])==7:
                team_dict[away][5][p][1].popleft()
                team_dict[away][5][p][1].append(0)
            else:
                team_dict[away][5][p][1].append(0)
        
        

        spin=1
        for inn in range(0,6):
           
            ## home pitching
            ### calculate li index before half inning starts
            home_defc=home_runs-away_runs

            ### picking relivers
            if inn<=5:
                
                new_p=pull_pitcher(team_dict,home,spinh,home_p,inn,away_runs,home_runs,rphome,away_runs_last_sub,home_last_inn_sub,1)
                #if new_p is None:
                
                if (spinh==1) and (new_p!=home_sp):
                    spinh=0
                    away_runs_last_sub=away_runs
                    home_last_inn_sub=inn
                    home_p=new_p
#                           findli(inn,home_defc,1,home,away,home_p,away_p,team_dict,0)
                    team_dict[home][7][home_p][3]+=1
                    rphome.remove(new_p)

                elif (spinh==0) and (new_p!=home_p):    
                    if len(team_dict[home][5][home_p][1])==7:
                        team_dict[home][5][home_p][1].popleft()
                        team_dict[home][5][home_p][1].append(inn-home_last_inn_sub)
                    else:
                        team_dict[home][5][home_p][1].append(inn-home_last_inn_sub)
                    away_runs_last_sub=away_runs
                    home_last_inn_sub=inn
                    home_p=new_p
                    
                    rphome.remove(new_p)
                    
                   # findli(inn,home_defc,1,home,away,home_p,away_p,team_dict,0)
                    team_dict[home][7][home_p][3]+=1


            li=findli(inn,home_defc,1,home,away,home_p,away_p,team_dict,1)
            team_dict[home][7][home_p][0]+=li
            runprob=team_dict[home][5][home_p][0]
            #runs=numpy.random.binomial(3,float(runprob)/float(3))
            runs=numpy.random.poisson(float(runprob),1)[0]
            away_runs+=runs
            

            team_dict[home][7][home_p][1]+=1
            team_dict[home][7][home_p][2]+=runs

            if inn==5:
                if home_p!=home_sp:
                    if len(team_dict[home][5][home_p][1])==7:
                        team_dict[home][5][home_p][1].popleft()
                        team_dict[home][5][home_p][1].append(6-home_last_inn_sub)
                    else:
                        team_dict[home][5][home_p][1].append(6-home_last_inn_sub)
               

                    
            if inn==5 and home_runs>away_runs:
                if away_p!=away_sp:
                    if len(team_dict[away][5][away_p][1])==7:
                        team_dict[away][5][away_p][1].popleft()
                        team_dict[away][5][away_p][1].append(5-away_last_inn_sub)
                    else:
                        team_dict[away][5][away_p][1].append(5-away_last_inn_sub)
                

                break


            ## away pitching

            away_defc=home_runs-away_runs

            ### picking relivers
            if inn<=5:
                new_p=pull_pitcher(team_dict,away,spina,away_p,inn,home_runs,away_runs,rpaway,home_runs_last_sub,away_last_inn_sub,1)
                if (spina==1) and (new_p!=away_sp):
                    spina=0
                    home_runs_last_sub=home_runs
                    away_last_inn_sub=inn
                    away_p=new_p
#                           findli(inn,away_defc,0,home,away,home_p,away_p,team_dict,0)
                    team_dict[away][7][away_p][3]+=1

                    rpaway.remove(new_p)
                    
                elif (spina==0) and (new_p!=away_p):
                    

                    if len(team_dict[away][5][away_p][1])==7:
                        team_dict[away][5][away_p][1].popleft()
                        team_dict[away][5][away_p][1].append(inn-away_last_inn_sub)
                    else:
                        team_dict[away][5][away_p][1].append(inn-away_last_inn_sub)
                    home_runs_last_sub=home_runs
                    away_last_inn_sub=inn
                    rpaway.remove(new_p)
                   
                    away_p=new_p
                    #findli(inn,away_defc,0,home,away,home_p,away_p,team_dict,0)
                    team_dict[away][7][away_p][3]+=1


            li=findli(inn,home_defc,0,home,away,home_p,away_p,team_dict,1)
            team_dict[away][7][away_p][0]+=li
            runprob=team_dict[away][5][away_p][0]
#                       runs=numpy.random.binomial(3,float(runprob)/float(3))
            runs=numpy.random.poisson(float(runprob),1)[0]

            home_runs+=runs

            team_dict[away][7][away_p][1]+=1
            team_dict[away][7][away_p][2]+=runs

            if inn==5:
                if away_p!=away_sp:
                    if len(team_dict[away][5][away_p][1])==7:
                        team_dict[away][5][away_p][1].popleft()
                        team_dict[away][5][away_p][1].append(5-away_last_inn_sub)
                    else:
                        team_dict[away][5][away_p][1].append(5-away_last_inn_sub)
                    
        #if i<0:
 #           print(rphome)
##                    print('game')
##                    print('--------')
##                    print('home')
##                    print('--------')
##                    print(home_runs)
##                    print('--------')
##                    print(away_runs)
##                    print('--------')
##                    print('end game')
        team_dict[home][0].append(home_sp)
        team_dict[away][0].append(away_sp)
        if home_runs>away_runs:
            team_dict[home][8][0]+=1
            team_dict[away][8][1]+=1
        else:
            team_dict[away][8][0]+=1
            team_dict[home][8][1]+=1
        for p in rphome:
            if len(team_dict[home][5][p][1])==7:
                team_dict[home][5][p][1].popleft()
                team_dict[home][5][p][1].append(0)
            else:
                team_dict[home][5][p][1].append(0)
       
        for p in rpaway:
            if len(team_dict[away][5][p][1])==7:
                team_dict[away][5][p][1].popleft()
                team_dict[away][5][p][1].append(0)
            else:
                team_dict[away][5][p][1].append(0)

    elif h==0 and a==1:
        home_sp=team_dict[home][0].popleft()
        home_p=home_sp
 
        ##home
        rphome=[10,11]
        homeremove=[]
                              
        
 
                              
        pavalaway=[0,10,11]
        away_removed=[]
        
##        for p in range(2,5):
##            if team_dict[away][5][p][1][6]>=4 or team_dict[away][5][p][1][5]>=4  or team_dict[away][5][p][1][4]>=4:
##                pavalaway.remove(p)
##                away_removed.append(p)
##            elif team_dict[away][5][p][1][6]>=3:
##                pavalaway.remove(p)
##                away_removed.append(p)
##            
##            elif team_dict[away][5][p][1][6]>0 and team_dict[away][5][p][1][5]>0 and team_dict[away][5][p][1][4]>0:
##                pavalaway.remove(p)
##                away_removed.append(p)
        for p in range(0,1):
            if team_dict[away][5][p][1][6]>=4 or team_dict[away][5][p][1][5]>=4  or team_dict[away][5][p][1][4]>=4:
                pavalaway.remove(p)
                away_removed.append(p)
            elif team_dict[away][5][p][1][6]>=3 or team_dict[away][5][p][1][5]>=3:
                pavalaway.remove(p)
                away_removed.append(p)
            elif team_dict[away][5][p][1][6]==2 and team_dict[away][5][p][1][5]==2:
                pavalaway.remove(p)
                away_removed.append(p)
            
            elif team_dict[away][5][p][1][6]>0 and team_dict[away][5][p][1][5]>0 and team_dict[away][5][p][1][4]>0:
                pavalaway.remove(p)
                away_removed.append(p)
                
##        for p in range(5,8):
##            if team_dict[away][5][p][1][6]>=2:
##                pavalaway.remove(p)
##                away_removed.append(p)
##            elif team_dict[away][5][p][1][6]>=1 and team_dict[away][5][p][1][5]>=2:
##                pavalaway.remove(p)
##                away_removed.append(p)
##            elif sum(team_dict[away][5][p][1])>4:
##                pavalaway.remove(p)
##                away_removed.append(p)
        for p in range(8,10):
            if team_dict[away][5][p][1][6]>=2 or team_dict[away][5][p][1][5]>=2:
                pavalaway.remove(p)
                away_removed.append(p)
            elif team_dict[away][5][p][1][6]>=1 and team_dict[away][5][p][1][5]>=1:
                pavalaway.remove(p)
                away_removed.append(p)
            elif sum(team_dict[away][5][p][1])>3:
                pavalaway.remove(p)
                away_removed.append(p)
            
        for p in range(10,11):
            if team_dict[away][5][p][1][6]>=4 or team_dict[away][5][p][1][5]>=4 :
                pavalaway.remove(p)
                away_removed.append(p)
            elif team_dict[away][5][p][1][6]>=3:
                pavalaway.remove(p)
                away_removed.append(p)
            elif team_dict[away][5][p][1][6]>0 and team_dict[away][5][p][1][5]>0 and team_dict[away][5][p][1][4]>1:
                pavalaway.remove(p)
                away_removed.append(p)

##        potsp=[]
        
##        for p in pavalaway:
##            if p==2 or p==3 or p==4:
##                potsp.append([p,2])
##    
##            
##        away_sp=picksp(team_dict,away,potsp,home_sp)
        away_sp=team_dict[away][9].popleft()
        away_sp=[away_sp,2]
        away_p=away_sp
        team_dict[away][9].append(away_sp[0])
        
        
 #       pavalaway.remove(away_sp[0])
        

##        runvals=[]
##        for p in pavalaway:
##            runvals.append(team_dict[away][5][p][0])
##        plist=sorted(zip(runvals,pavalaway), reverse=False)
##        away_high_lev=[]
##                ## desginate 3 (or rest) of pitchers for med lev situations
##
##        away_med_lev=[]
##
##        ## designate rest for low lev situations
##        away_low_lev=[]
##        away_plist=[]
##
##
##        for ii in range(0,len(plist)):
##            away_plist.append(plist[ii][1])
##            if  plist[ii][1]==0 or  plist[ii][1]==1:
##                continue
##            if ii<=2:
##                away_high_lev.append([plist[ii][1],3])
##            elif ii>2 and ii<=6:
##                away_med_lev.append([plist[ii][1],2])
##            else:
##               away_low_lev.append([plist[ii][1],1])
##        
##        away_high_lev_sp=[]
##        if 0 in pavalaway:
##            away_high_lev_sp.append([0,3])
##        if 1 in pavalaway:
##            away_high_lev_sp.append([1,3])
        
        #print('before')

        
        #print('after')
       
        
        for p in away_removed:
            

            if len(team_dict[away][5][p][1])==7:
                team_dict[away][5][p][1].popleft()
                team_dict[away][5][p][1].append(0)
            else:
                team_dict[away][5][p][1].append(0)

        ### home is normal
        away_high_lev_sp=[]
        away_low_lev=[]
                
        if 0 in pavalaway:
            away_high_lev_sp.append([0,3])
        if 1 in pavalaway:
            away_high_lev_sp.append([1,3])
        if 10 in pavalaway:
            away_low_lev.append([10,1])
        if 11 in pavalaway:
            away_low_lev.append([11,1])
                
        used=[]
        for inn in range(0,6):
           
            ## home pitching
            ### calculate li index before half inning starts
            home_defc=home_runs-away_runs

            ### picking relivers
            if inn<=8:
                
                new_p=pull_pitcher(team_dict,home,spinh,home_p,inn,away_runs,home_runs,rphome,away_runs_last_sub,home_last_inn_sub,1)
                #if new_p is None:
                
                    
                if (spinh==1) and (new_p!=home_sp):
                    spinh=0
                    away_runs_last_sub=away_runs
                    home_last_inn_sub=inn
                    home_p=new_p
#                           findli(inn,home_defc,1,home,away,home_p,away_p,team_dict,0)
                    team_dict[home][7][home_p][3]+=1
                    rphome.remove(new_p)

                elif (spinh==0) and (new_p!=home_p):
                    if len(team_dict[home][5][home_p][1])==7:
                        team_dict[home][5][home_p][1].popleft()
                        team_dict[home][5][home_p][1].append(inn-home_last_inn_sub)
                    else:
                        team_dict[home][5][home_p][1].append(inn-home_last_inn_sub)
                    away_runs_last_sub=away_runs
                    home_last_inn_sub=inn
                    home_p=new_p
                    
                    rphome.remove(new_p)
                    
                   # findli(inn,home_defc,1,home,away,home_p,away_p,team_dict,0)
                    team_dict[home][7][home_p][3]+=1


            li=findli(inn,home_defc,1,home,away,home_p,away_p,team_dict,1)
            team_dict[home][7][home_p][0]+=li
            runprob=team_dict[home][5][home_p][0]
            #runs=numpy.random.binomial(3,float(runprob)/float(3))
            runs=numpy.random.poisson(float(runprob),1)[0]
            away_runs+=runs
            

            team_dict[home][7][home_p][1]+=1
            team_dict[home][7][home_p][2]+=runs

            if inn==5:
                if home_p!=home_sp:
                    if len(team_dict[home][5][home_p][1])==7:
                        team_dict[home][5][home_p][1].popleft()
                        team_dict[home][5][home_p][1].append(6-home_last_inn_sub)
                    else:
                        team_dict[home][5][home_p][1].append(6-home_last_inn_sub)
            if inn==5 and home_runs>away_runs:
                if len(team_dict[away][5][away_p[0]][1])==7:
                    team_dict[away][5][away_p[0]][1].popleft()
                    team_dict[away][5][away_p[0]][1].append(5-away_last_inn_sub)
                else:
                    team_dict[away][5][away_p[0]][1].append(5-away_last_inn_sub)
                break
            

            ## away pitching
            if inn<=5:
                li=findli(inn,home_defc,0,home,away,home_p,away_p,team_dict,1)
                
                new_p=pull_pitcher2(team_dict,away,spina,away_p,inn,home_runs,away_runs,away_low_lev,home_runs_last_sub,away_last_inn_sub,li,away_high_lev_sp)

                if new_p==-1 or new_p is None:
                    team_dict[away][0].append(away_sp[0])
                    print('0000--away')
                    return([team_dict,away,away_p,inn,li,away_high_lev,away_med_lev,away_low_lev])
                if (spina==1) and (new_p!=away_sp):
                    used.append(new_p[0])

                    if len(team_dict[away][5][away_p[0]][1])==7:
                        team_dict[away][5][away_p[0]][1].popleft()
                        team_dict[away][5][away_p[0]][1].append(inn-away_last_inn_sub)
                    else:
                        team_dict[away][5][away_p[0]][1].append(inn-away_last_inn_sub)
                    spina=0
                    home_runs_last_sub=home_runs
                    away_last_inn_sub=inn
                    away_p=new_p
##                    if new_p in away_high_lev and new_p in away_high_lev_sp :
##                        away_high_lev.remove(new_p)
##                        away_high_lev_sp.remove(new_p)
##                    elif new_p in away_high_lev:
##                        away_high_lev.remove(new_p)
##                    elif new_p in away_high_lev_sp:
##                        away_high_lev_sp.remove(new_p)
##                    elif new_p in away_med_lev:
##                        away_med_lev.remove(new_p)
##                    elif new_p in away_low_lev:
##                        away_low_lev.remove(new_p)
##                    elif new_p in away_high_lev_sp:
##                        away_low_lev.remove(new_p)
                    
                elif (spina==0) and (new_p!=away_p):
                    used.append(new_p[0])

                    
                    if len(team_dict[away][5][away_p[0]][1])==7:
                        team_dict[away][5][away_p[0]][1].popleft()
                        team_dict[away][5][away_p[0]][1].append(inn-away_last_inn_sub)
                    else:
                        team_dict[away][5][away_p[0]][1].append(inn-away_last_inn_sub)
                    home_runs_last_sub=home_runs
                    away_last_inn_sub=inn
                    away_p=new_p
##                    if new_p in away_high_lev and new_p in away_high_lev_sp :
##                        away_high_lev.remove(new_p)
##                        away_high_lev_sp.remove(new_p)
##                    elif new_p in away_high_lev:
##                        away_high_lev.remove(new_p)
##                    elif new_p in away_high_lev_sp:
##                        away_high_lev_sp.remove(new_p)
##                    elif new_p in away_med_lev:
##                        away_med_lev.remove(new_p)
##                    elif new_p in away_low_lev:
##                        away_low_lev.remove(new_p)
##                    elif new_p in away_high_lev_sp:
##                        away_low_lev.remove(new_p)


            away_defc=home_runs-away_runs
            li=findli(inn,home_defc,0,home,away,home_p,away_p,team_dict,1)
            team_dict[away][7][away_p[0]][0]+=li
            
            runprob=team_dict[away][5][away_p[0]][0]
##            if away_p[0]==8 or away_p[0]==9:
##                runprob+=.3
            runs=numpy.random.poisson(float(runprob),1)[0]

            home_runs+=runs

            team_dict[away][7][away_p[0]][1]+=1
            team_dict[away][7][away_p[0]][2]+=runs
            if inn==5:
                if len(team_dict[away][5][away_p[0]][1])==7:
                    team_dict[away][5][away_p[0]][1].popleft()
                    team_dict[away][5][away_p[0]][1].append(6-away_last_inn_sub)
                else:
                    team_dict[away][5][away_p[0]][1].append(6-away_last_inn_sub)
                break
        team_dict[home][0].append(home_sp)
        if home_runs>away_runs:
            team_dict[home][8][0]+=1
            team_dict[away][8][1]+=1
        else:
            team_dict[away][8][0]+=1
            team_dict[home][8][1]+=1
        track=[]
##        for p in away_high_lev:
##            track.append(p)
##            if len(team_dict[away][5][p[0]][1])==7:
##                team_dict[away][5][p[0]][1].popleft()
##                team_dict[away][5][p[0]][1].append(0)
##            else:
##                team_dict[away][5][p[0]][1].append(0)
##        for p in away_high_lev_sp:
##            if p in track:
##                continue
##            if len(team_dict[away][5][p[0]][1])==7:
##                team_dict[away][5][p[0]][1].popleft()
##                team_dict[away][5][p[0]][1].append(0)
##            else:
##                team_dict[away][5][p[0]][1].append(0)
##        for p in away_med_lev:
##            if len(team_dict[away][5][p[0]][1])==7:
##                team_dict[away][5][p[0]][1].popleft()
##                team_dict[away][5][p[0]][1].append(0)
##            else:
##                team_dict[away][5][p[0]][1].append(0)
##        for p in away_low_lev:
##            if len(team_dict[away][5][p[0]][1])==7:
##                team_dict[away][5][p[0]][1].popleft()
##                team_dict[away][5][p[0]][1].append(0)
##            else:
##                team_dict[away][5][p[0]][1].append(0)
        if 0 not in used:
            team_dict[away][5][0][1].popleft()
            team_dict[away][5][0][1].append(0)
##        if 1 not in used:
##            team_dict[away][5][1][1].popleft()
##            team_dict[away][5][1][1].append(0)
        if 10 not in used:
            team_dict[away][5][10][1].popleft()
            team_dict[away][5][10][1].append(0)
        if 11 not in used:
            team_dict[away][5][11][1].popleft()
            team_dict[away][5][11][1].append(0)
            

        for p in rphome:
            if len(team_dict[home][5][p][1])==7:
                team_dict[home][5][p][1].popleft()
                team_dict[home][5][p][1].append(0)
            else:
                team_dict[home][5][p][1].append(0)


       
    elif h==1 and a==0:
 #       print 'inn || home_runs || away_runs || li || pitcher'
        
        spinh=1

        away_runs_last_sub=0
        home_inn_last_sub=0

        away_runs=0

        ##home
 

        rpaway=[10,11]
        awayremove=[]
        away_sp=team_dict[away][0].popleft()
        away_p=away_sp
        

 
        


        ### add in away normal here
                #####
                #####
            

        spinh=1
        spina=1
        home_last_inn_sub=0
        away_last_inn_sub=0

        away_runs_last_sub=0
        home_runs_last_sub=0

##        if i<0:
##           
##            print('------------')
##            print(home_high_lev)
##            print(home_med_lev)
##            print(home_low_lev)
##            print('------------')
##        if show==2:
##            print('game')
##            print(i)
##            print('------------')
##            print(home_high_lev)
##            print(home_med_lev)
##            print(home_low_lev)
##            print('----------')
##            print('----------')
##            print('----------')


        pavalhome=[0,10,11]
        home_removed=[]
        
##        for p in range(2,5):
##            if team_dict[home][5][p][1][6]>=4 or team_dict[home][5][p][1][5]>=4  or team_dict[home][5][p][1][4]>=4:
##                pavalhome.remove(p)
##                home_removed.append(p)
##            elif team_dict[home][5][p][1][6]>=3:
##                pavalhome.remove(p)
##                home_removed.append(p)
##            elif team_dict[home][5][p][1][6]>0 and team_dict[home][5][p][1][5]>0 and team_dict[home][5][p][1][4]>0:
##                pavalhome.remove(p)
##                home_removed.append(p)
        for p in range(0,1):
            if team_dict[home][5][p][1][6]>=4 or team_dict[home][5][p][1][5]>=4  or team_dict[home][5][p][1][4]>=4:
                pavalhome.remove(p)
                home_removed.append(p)
            elif team_dict[home][5][p][1][6]>=3 or team_dict[home][5][p][1][5]>=3:
                pavalhome.remove(p)
                home_removed.append(p)
            elif team_dict[home][5][p][1][6]==2 and team_dict[home][5][p][1][5]==2:
                pavalhome.remove(p)
                home_removed.append(p)
            elif team_dict[home][5][p][1][6]>0 and team_dict[home][5][p][1][5]>0 and team_dict[home][5][p][1][4]>0:
                pavalhome.remove(p)
                home_removed.append(p)
                
##        for p in range(5,8):
##            if team_dict[home][5][p][1][6]>=2:
##                pavalhome.remove(p)
##                home_removed.append(p)
##            elif team_dict[home][5][p][1][6]>=1 and team_dict[home][5][p][1][5]>=2:
##                pavalhome.remove(p)
##                home_removed.append(p)
##            elif sum(team_dict[home][5][p][1])>4:
##                pavalhome.remove(p)
##                home_removed.append(p)
            
        for p in range(8,10):
            if team_dict[home][5][p][1][6]>=2 or team_dict[home][5][p][1][5]>=2:
                pavalhome.remove(p)
                home_removed.append(p)
            elif team_dict[home][5][p][1][6]>=1 and team_dict[home][5][p][1][5]>=1:
                pavalhome.remove(p)
                home_removed.append(p)
            elif sum(team_dict[home][5][p][1])>3:
                pavalhome.remove(p)
                home_removed.append(p)
            
        for p in range(10,11):
            if team_dict[home][5][p][1][6]>=4 or team_dict[home][5][p][1][5]>=4:
                pavalhome.remove(p)
                home_removed.append(p)
            elif team_dict[home][5][p][1][6]>=3:
                pavalhome.remove(p)
                home_removed.append(p)
            elif team_dict[home][5][p][1][6]>0 and team_dict[home][5][p][1][5]>0 and team_dict[home][5][p][1][4]>1:
                pavalhome.remove(p)
                home_removed.append(p)
        

##        potsp=[]
##        for p in pavalhome:
##            if p==2 or p==3 or p==4:
##                potsp.append([p,2])
##            
##        home_sp=picksp(team_dict,home,potsp,away_sp)

        home_sp=team_dict[home][9].popleft()
        home_sp=[home_sp,2]
        home_p=home_sp
        team_dict[home][9].append(home_sp[0])
        
        
 #       pavalhome.remove(home_sp[0])
       

##        runvals=[]
##        for p in pavalhome:
##            runvals.append(team_dict[home][5][p][0])
##        plist=sorted(zip(runvals,pavalhome), reverse=False)
##        home_high_lev=[]
##                ## desginate 3 (or rest) of pitchers for med lev situations
##
##        home_med_lev=[]
##
##        ## designate rest for low lev situations
##        home_low_lev=[]
##        home_plist=[]
##
##
##        for ii in range(0,len(plist)):
##            if plist[ii][1]==1 or plist[ii][1]==0 :
##                continue
##            home_plist.append(plist[ii][1])
##            if ii<=2:
##                home_high_lev.append([plist[ii][1],3])
##            elif ii>2 and ii<=6:
##                home_med_lev.append([plist[ii][1],2])
##            else:
##               home_low_lev.append([plist[ii][1],1])
##        
##        home_high_lev_sp=[]
##        if 0 in pavalhome:
##            home_high_lev_sp.append([0,3])
##        if 1 in pavalhome:
##            home_high_lev_sp.append([1,3])
##        

        
        #print('before')

        
        #print('after')
        #print(home_med_lev)
        
        
        home_p=home_sp
        for p in home_removed:

            if len(team_dict[home][5][p][1])==7:
                team_dict[home][5][p][1].popleft()
                team_dict[home][5][p][1].append(0)
            else:
                team_dict[home][5][p][1].append(0)



        home_high_lev_sp=[]
        home_low_lev=[]
                
        if 0 in pavalhome:
            home_high_lev_sp.append([0,3])
##        if 1 in pavalhome:
##            home_high_lev_sp.append([1,3])
        if 10 in pavalhome:
            home_low_lev.append([10,1])
        if 11 in pavalhome:
            home_low_lev.append([11,1])
        used=[]
                
##        
        for inn in range(0,6):
##            print('inn')
##            print(inn)
##            
##            print('home score')
##            print(home_runs)
##            print('away score')
##            print(away_runs)
##            print('pitcher')
            
            
            ## home pitching
            ### calculate li index before half inning starts
            home_defc=home_runs-away_runs

            if inn<=5:
##                if show==2:
##                    
##                    print('inn')
##                    print('--------')
##                    print(inn)
##                    print('score')
##                    print('--------')
##                    print(home_runs)
##                    print(away_runs)
##                    print('--------')
                li=findli(inn,home_defc,1,home,away,home_p,away_p,team_dict,1)
##                        print('li')
##                        print(li)
##                        print('--------')

                new_p=pull_pitcher2(team_dict,home,spinh,home_p,inn,away_runs,home_runs,home_low_lev,away_runs_last_sub,home_last_inn_sub,li,home_high_lev_sp)
 #               print inn,'  ||     ',home_runs,'   ||     ', away_runs,'   ||',li,'||     ',new_p[0]
##                if show==2:
##                    print(new_p)
##                    print('end inning')
                
                if new_p==-1 or new_p is None:
                    team_dict[home][0].append(home_sp[0])
                    
 #                   print(home_removed)
                    print('0000--home')
                    return([team_dict,home,home_p,inn,li,home_high_lev,home_med_lev,home_low_lev])
                if (spinh==1) and (new_p!=home_sp):
                    used.append(new_p[0])
                    if len(team_dict[home][5][home_p[0]][1])==7:
                        
                        team_dict[home][5][home_p[0]][1].popleft()
                        team_dict[home][5][home_p[0]][1].append(inn-home_last_inn_sub)
                        
                    else:
                        team_dict[home][5][home_p[0]][1].append(inn-home_last_inn_sub)
                    

                    spinh=0
                    away_runs_last_sub=away_runs
                    home_last_inn_sub=inn
                    
                    home_p=new_p
##                    if new_p in home_high_lev and new_p in home_high_lev_sp :
##                        home_high_lev.remove(new_p)
##                        home_high_lev_sp.remove(new_p)
##                    elif new_p in home_high_lev:
##                        home_high_lev.remove(new_p)
##                    elif new_p in home_high_lev_sp:
##                        home_high_lev_sp.remove(new_p)
##                    elif new_p in home_med_lev:
##                        home_med_lev.remove(new_p)
##                    elif new_p in home_low_lev:
##                        home_low_lev.remove(new_p)
##                    elif new_p in home_high_lev_sp:
##                        home_low_lev.remove(new_p)
                    
                    
                elif (spinh==0) and (new_p!=home_p):
                    used.append(new_p[0])

                    if len(team_dict[home][5][home_p[0]][1])==7:                                    
                        
                        team_dict[home][5][home_p[0]][1].popleft()
                        team_dict[home][5][home_p[0]][1].append(inn-home_last_inn_sub)
                        
                    else:
                        
                        
                        team_dict[home][5][home_p[0]][1].append(inn-home_last_inn_sub)
                        
                    away_runs_last_sub=away_runs
                    home_last_inn_sub=inn
                   
                    home_p=new_p
##                    if new_p in home_high_lev and new_p in home_high_lev_sp :
##                        home_high_lev.remove(new_p)
##                        home_high_lev_sp.remove(new_p)
##                    elif new_p in home_high_lev:
##                        home_high_lev.remove(new_p)
##                    elif new_p in home_high_lev_sp:
##                        home_high_lev_sp.remove(new_p)
##                    elif new_p in home_med_lev:
##                        home_med_lev.remove(new_p)
##                    elif new_p in home_low_lev:
##                        home_low_lev.remove(new_p)
##                    elif new_p in home_high_lev_sp:
##                        home_low_lev.remove(new_p)
            
            li=findli(inn,home_defc,1,home,away,home_p,away_p,team_dict,1)
            team_dict[home][7][home_p[0]][0]+=li
           

            
#               runs=howmanyruns()
            
            runprob=team_dict[home][5][home_p[0]][0]
##            if home_p[0]==8 or home_p[0]==9:
##                runprob+=.3
                #runs=numpy.random.binomial(3,float(runprob)/float(3))
            runs=numpy.random.poisson(float(runprob),1)[0]

            away_runs+=runs
            

            team_dict[home][7][home_p[0]][1]+=1
            team_dict[home][7][home_p[0]][2]+=runs


            home_defc=home_runs-away_runs

            if inn==5:
                if len(team_dict[home][5][home_p[0]][1])==7:
                    team_dict[home][5][home_p[0]][1].popleft()
                    team_dict[home][5][home_p[0]][1].append(6-home_last_inn_sub)
                else:
                    team_dict[home][5][home_p[0]][1].append(6-home_last_inn_sub)


    

            if inn==5 and home_runs>away_runs:
                if away_p!=away_sp:
                    if len(team_dict[away][5][away_p][1])==7:
                        team_dict[away][5][away_p][1].popleft()
                        team_dict[away][5][away_p][1].append(5-away_last_inn_sub)
                    else:
                        team_dict[away][5][away_p][1].append(5-away_last_inn_sub)
                

                break


            ## away pitching

            away_defc=home_runs-away_runs

            ### picking relivers
            if inn<=5:
                new_p=pull_pitcher(team_dict,away,spina,away_p,inn,home_runs,away_runs,rpaway,home_runs_last_sub,away_last_inn_sub,1)
                if (spina==1) and (new_p!=away_sp):
                    spina=0
                    home_runs_last_sub=home_runs
                    away_last_inn_sub=inn
                    away_p=new_p
#                           findli(inn,away_defc,0,home,away,home_p,away_p,team_dict,0)
                    team_dict[away][7][away_p][3]+=1

                    rpaway.remove(new_p)
                    
                elif (spina==0) and (new_p!=away_p):
                    

                    if len(team_dict[away][5][away_p][1])==7:
                        team_dict[away][5][away_p][1].popleft()
                        team_dict[away][5][away_p][1].append(inn-away_last_inn_sub)
                    else:
                        team_dict[away][5][away_p][1].append(inn-away_last_inn_sub)
                    home_runs_last_sub=home_runs
                    away_last_inn_sub=inn
                    rpaway.remove(new_p)
                   
                    away_p=new_p
                    #findli(inn,away_defc,0,home,away,home_p,away_p,team_dict,0)
                    team_dict[away][7][away_p][3]+=1


            li=findli(inn,home_defc,0,home,away,home_p,away_p,team_dict,1)
            team_dict[away][7][away_p][0]+=li
            runprob=team_dict[away][5][away_p][0]
#                       runs=numpy.random.binomial(3,float(runprob)/float(3))
            runs=numpy.random.poisson(float(runprob),1)[0]

            home_runs+=runs

            team_dict[away][7][away_p][1]+=1
            team_dict[away][7][away_p][2]+=runs

            if inn==5:
                if away_p!=away_sp:
                    if len(team_dict[away][5][away_p][1])==7:
                        team_dict[away][5][away_p][1].popleft()
                        team_dict[away][5][away_p][1].append(6-away_last_inn_sub)
                    else:
                        team_dict[away][5][away_p][1].append(6-away_last_inn_sub)
                    

            
            
                
                

            
                    
##        if i<0:
##            print('game')
##            print('--------')
##            print('home')
##            print('--------')
##            print(home_runs)
##            print('--------')
##            print(away_runs)
##            print('--------')
##            print('end game')
        team_dict[away][0].append(away_sp)

        if home_runs>away_runs:
            team_dict[home][8][0]+=1
            team_dict[away][8][1]+=1
        else:
            team_dict[away][8][0]+=1
            team_dict[home][8][1]+=1

##        track=[]
##        
##        for p in home_high_lev:
##            track.append(p)
##            if len(team_dict[home][5][p[0]][1])==7:
##                team_dict[home][5][p[0]][1].popleft()
##                team_dict[home][5][p[0]][1].append(0)                      
##            else:
##                team_dict[home][5][p[0]][1].append(0)
##                
##        for p in home_med_lev:
##            if len(team_dict[home][5][p[0]][1])==7:
##   
##                team_dict[home][5][p[0]][1].popleft()
##                team_dict[home][5][p[0]][1].append(0)                       
##            else:
##                team_dict[home][5][p[0]][1].append(0)
##                
##        for p in home_low_lev:
##            if len(team_dict[home][5][p[0]][1])==7: 
##                team_dict[home][5][p[0]][1].popleft()
##                team_dict[home][5][p[0]][1].append(0)                       
##            else: 
##                team_dict[home][5][p[0]][1].append(0)
##        for p in home_high_lev_sp:
##            if p in track:
##                continue
##            if len(team_dict[home][5][p[0]][1])==7: 
##                team_dict[home][5][p[0]][1].popleft()
##                team_dict[home][5][p[0]][1].append(0)                       
##            else: 
##                team_dict[home][5][p[0]][1].append(0)
        
##
        if 0 not in used:
            team_dict[away][5][0][1].popleft()
            team_dict[away][5][0][1].append(0)
##        if 1 not in used:
##            team_dict[away][5][1][1].popleft()
##            team_dict[away][5][1][1].append(0)
        if 10 not in used:
            team_dict[away][5][10][1].popleft()
            team_dict[away][5][10][1].append(0)
        if 11 not in used:
            team_dict[away][5][11][1].popleft()
            team_dict[away][5][11][1].append(0)

        for p in rpaway:
            if len(team_dict[away][5][p][1])==7:
                team_dict[away][5][p][1].popleft()
                team_dict[away][5][p][1].append(0)
            else:
                team_dict[away][5][p][1].append(0)
                

    return(team_dict)
                
        
                    
 
                        
            
            
def pull_pitcher(team_dict,team,spin,cur_p,inn,oppruns,runs,rpaval,home_runs_last_sub,inn_last_sub,two):
    import itertools
            ## conditions to pull pitcher and what pithcer to bring in
            ## use model for sp
            ## for relivers, create conditions
    if two==0:
        if spin==1:
            if pullstarter(inn,runs):
                if len(rpaval)==1:
                    return(rpaval[0])
                if len(rpaval)==0:
                    return(cur_p)                   
                if sum(team_dict[team][1][10][1])<sum(team_dict[team][1][11][1]):
                    return(10)
                else:
                    return(11)
            else:
                return(cur_p)
        else:
            if len(rpaval)==1:
                return(rpaval[0])
            if len(rpaval)==0:
                return(cur_p)                   
            if sum(team_dict[team][1][10][1])<sum(team_dict[team][1][11][1]):
                return(10)
            else:
                return(11)
            
    else:
        if spin==1:
            if pullstarter(inn,runs):
                if len(rpaval)==1:
                    return(rpaval[0])
                if len(rpaval)==0:
                    return(cur_p)
                if sum(team_dict[team][5][10][1])<sum(team_dict[team][5][11][1]):
                    return(10)
                else:
                    return(11)
            else:
                return(cur_p)
        else:
            if len(rpaval)==1:
                return(rpaval[0])
            if len(rpaval)==0:
                return(cur_p)
            if sum(team_dict[team][5][10][1])<sum(team_dict[team][5][11][1]):
                return(10)
            else:
                return(11)
            
                                
   
                                                   
                   
                    
def pullstarter(inn,runs):
    if inn==1 and runs>5:
        return(True)
    elif inn==2 and runs>4:
        return(True)
    elif inn==3 and runs>4:
        return(True)
    elif inn==4 and runs>4:
        return(True)
    elif inn==5 and runs>3:
        return(True)
    elif inn==6 and runs>2:
        return(True)
    elif inn==7 and runs>2:
        return(True)
    elif inn==8 and runs>0:
        return(True)
    else:
        return(False)
                    
                        

### function to check if current mop up guy should be taken out
                        ## true= take out
def mopupcheck(cur_p,team_dict,team,inn,defc,runs_given_up,gameip,rpaval):
    pastip=sum(team_dict[team][1][cur_p][1])
    if inn<=6 and len(rpaval)<=3:
        return(False)
    elif inn>6 and len(rpaval)<2:
        if defc>3 or defc<-1:
            return(False)
    if pastip+gameip>6:
        return(True)
    if cur_p<=8:
        if gameip==1:
            if (runs_given_up==0) or (runs_given_up<2 and pastip+gameip<3):
                return(False)
            else:
                return(True)
        else:
            return(True)
    
    elif cur_p>8:
        if pastip+gameip>3:
            return(True)
        elif gameip>2:
            if runs_given_up==0:
                return(False)
            else:
                return(True)
        else:
            return(False)
    
                                           
            





def pullrest(team_dict,team,cur_p,inn,runs_given_up,innp,lev,high_lev_sp,allp,li):
    pastip=(team_dict[team][5][cur_p[0]][1])
    if innp==0:
        return(False)
    if len(allp)==0 and li<1.1:
        return(False)
    elif len(high_lev_sp)==0 and li>1:
        return(False)
    
    if  cur_p[0]<1:
        if innp==1:
            if runs_given_up>=5:
                return(True)
            elif team_dict[team][5][cur_p[0]][1][6]>3:
                return(True)
            else:
                return(False)
        elif innp==2:
            if runs_given_up>=4:
                return(True)
            elif team_dict[team][5][cur_p[0]][1][6]>0 and team_dict[team][5][cur_p[0]][1][5]>0:
                return(True)
            elif team_dict[team][5][cur_p[0]][1][4]>=3:
                return(True)
            else:
                return(False)
        elif innp==3:
            if runs_given_up>=3:
                return(True)
            
            elif team_dict[team][5][cur_p[0]][1][6]>0 or team_dict[team][5][cur_p[0]][1][5]>2 or team_dict[team][5][cur_p[0]][1][4]>1:
                return(True)
            else:
                return(False)
        elif innp==4:
            if runs_given_up>=1:
                return(True)
            
            elif team_dict[team][5][cur_p[0]][1][6]>0 or team_dict[team][5][cur_p[0]][1][5]>0 or team_dict[team][5][cur_p[0]][1][4]>0:
                return(True)
            else:
                return(False)
            
        elif innp>=5:
            return(True)
    elif cur_p[0]==2 or cur_p[0]==1 or cur_p[0]==3 or cur_p[0]==4:
        if innp<=2:
            return(False)
        if innp==3:
            if runs_given_up>=3:
                return(True)
            elif team_dict[team][5][cur_p[0]][1][6]>=4:
                return(True)
            else:
                return(False)
            return(False)
        elif innp==4:
            if runs_given_up>=2:
                return(True)
            elif team_dict[team][5][cur_p[0]][1][6]>=3:
                return(True)
            else:
                return(False)
        
            
        
    elif cur_p[0]>=10 :
        if innp==1:
            if runs_given_up>=4:
                return(True)
            elif team_dict[team][5][cur_p[0]][1][6]>2:
                return(True)
            else:
                return(False)
        elif innp==2:
            if runs_given_up>=3:
                return(True)
            elif team_dict[team][5][cur_p[0]][1][6]>0 and team_dict[team][5][cur_p[0]][1][5]>0 and team_dict[team][5][cur_p[0]][1][4]>0:
                return(True)
            elif team_dict[team][5][cur_p[0]][1][4]>=3:
                return(True)
            else:
                return(False)
        elif innp==3:
            
            if runs_given_up>=2:
                return(True)
            elif team_dict[team][5][cur_p[0]][1][6]>0 and team_dict[team][5][cur_p[0]][1][5]>0 and team_dict[team][5][cur_p[0]][1][4]>0:
                return(True)
            elif team_dict[team][5][cur_p[0]][1][4]>=3:
                return(True)
            else:
                return(False)
        elif innp>=4:
            return(True)
##    elif cur_p[0]==8 or cur_p[0]==9:
##        if innp==1:
##            if runs_given_up>=3:
##                return(True)
##            elif team_dict[team][5][cur_p[0]][1][6]>0:
##                return(True)
##            elif team_dict[team][5][cur_p[0]][1][5]>0:
##                return(True)
##            elif team_dict[team][5][cur_p[0]][1][4]>1:
##                return(True)
##            else:
##                return(False)
##        elif innp==2:
##            if runs_given_up>=2:
##                return(True)
##            elif team_dict[team][5][cur_p[0]][1][6]>0 or team_dict[team][5][cur_p[0]][1][5]>0 or team_dict[team][5][cur_p[0]][1][4]>0 :
##                return(True)
##            elif team_dict[team][5][cur_p[0]][1][3]>2:
##                return(True)
##            else:
##                return(False)
##        elif innp>=3:
##            return(True)
    else:
        if innp==1:
            if runs_given_up>=3:
                return(True)
            elif team_dict[team][5][cur_p[0]][1][6]>0 or team_dict[team][5][cur_p[0]][1][5]>0 :
                return(True)
            else:
                return(False)
        elif innp>=2:
            return(True)

##lev= what lev is the current p from
def choose_new_p(pulled_from_rest,team_dict,team,cur_p,inn,li,low_lev,high_lev_sp):
    pastip=team_dict[team][5][cur_p[0]][1]
    allp=high_lev_sp+low_lev

    if pulled_from_rest==0:
        if inn<=2:
            return(cur_p)
        minip=100
        minp=-1
        if inn==4 or inn==5:
            if li>=1.1:
                if len(high_lev_sp)>0:
                    return(high_lev_sp[0])
                else:
                    return(cur_p)
            else:
                return(cur_p)
##            if li>=1:
##                if cur_p[0]==1:
##                    return(cur_p)
##                else:
##                    if len(high_lev_sp)>0:
##                        return(high_lev_sp[0])
                
            
##        elif inn==3:
##            if li>=1.1:
##                if len(high_lev_sp)>0:
##                    return(high_lev_sp[0])
##                else:
##                    return(cur_p)
##            else:
##                return(cur_p)
        else:
            return(cur_p)
        
##                    minip=100
##                    minp=-1
##                    if len(allp)>0:
##                        for p in allp:
##                            if sum(team_dict[team][5][p[0]][1])<minip:
##                                minip=sum(team_dict[team][5][p[0]][1])
##                                minp=p
##                        return(minp)
##                    else:
##                        return(cur_p)
##                elif inn==5 and (cur_p[0]==0 or cur_p[0]==1):
##                    minip=100
##                    minp=-1
##                    if len(allp)>0:
##                        for p in allp:
##                            if sum(team_dict[team][5][p[0]][1])<minip:
##                                minip=sum(team_dict[team][5][p[0]][1])
##                                minp=p
##                        return(minp)
##                    else:
##                        return(cur_p)
##                else:                   
##                    return(cur_p)
##            
##        elif inn==3:
##            if li>=1.1 and cur_p[0]!=0 and cur_p[0]!=1:
##                
##                if len(high_lev_sp)>1:
##                    return(high_lev_sp[1])
##                elif len(high_lev_sp)==1:
##                    return(high_lev_sp[0])
##                else:
 ##                   return(cur_p)
##            elif li>=.9:
##                if len(high_lev_sp)==2:
##                    return(high_lev_sp[1])
##                elif len(high_lev_sp)==1 and high_lev_sp[0][0]==1:
##                    return(high_lev_sp[0])
##                else:
##                    return(cur_p)
        
           
##        elif inn==2:
##            if li>=1:
##                
##                if len(high_lev_sp)>1:
##                    return(high_lev_sp[1])
##                elif len(high_lev_sp)==1:
##                    return(high_lev_sp[0])
##                else:
##                    return(cur_p)
##            elif li>=.9:
##                if len(high_lev_sp)==2:
##                    return(high_lev_sp[1])
##                elif len(high_lev_sp)==1 and high_lev_sp[0][0]==1:
##                    return(high_lev_sp[0])
##                else:
##                    return(cur_p)
        
##            else:
##                return(cur_p)
##            
            
                
    else:
        minip=100
        minp=-1
        
        if inn<2:
            minip=100
            minp=-1
            if len(allp)>0:
                for p in allp:
                    if sum(team_dict[team][5][p[0]][1])<minip:
                        minip=sum(team_dict[team][5][p[0]][1])
                        minp=p
                return(minp)
            else:
                return(cur_p)       
        if inn==4 or inn==5:
            if li>=1:
                if len(high_lev_sp)>0:
                    return(high_lev_sp[0])
                else:
                    if len(low_lev)>0:
                        for p in low_lev:
                            if sum(team_dict[team][5][p[0]][1])<minip:
                                minip=sum(team_dict[team][5][p[0]][1])
                                minp=p
                        return(minp)
                    else:
                        return(cur_p) 
##            elif li>=1:
##                if len(high_lev_sp)==2:
##                    return(high_lev_sp[1])
##                elif len(high_lev_sp)==1 and high_lev_sp[0][0]==1:
##                    return(high_lev_sp[0])
##                else:
##                    minip=100
##                    minp=-1
##                    for p in allp:
##                        if sum(team_dict[team][5][p[0]][1])<minip:
##                            minip=sum(team_dict[team][5][p[0]][1])
##                            minp=p
##                    return(minp)
        
            else:
                
                if len(low_lev)>0:
                    for p in low_lev:
                        if sum(team_dict[team][5][p[0]][1])<minip:
                            minip=sum(team_dict[team][5][p[0]][1])
                            minp=p
                    return(minp)
                else:
                    return(cur_p)
        elif inn==3:
            if li>=1.1:
                if len(high_lev_sp)>0:
                    return(high_lev_sp[0])
                else:
                    return(cur_p)
##                    if len(low_lev)>0:
##                        for p in allp:
##                            if sum(team_dict[team][5][p[0]][1])<minip:
##                                minip=sum(team_dict[team][5][p[0]][1])
##                                minp=p
##                        return(minp)
##                    else:
##                        return(cur_p)
##            elif li>=.9:
##                if len(high_lev_sp)==2:
##                    return(high_lev_sp[1])
##                elif len(high_lev_sp)==1 and high_lev_sp[0][0]==1:
##                    return(high_lev_sp[0])
##                else:
##                    minip=100
##                    minp=-1
##                    for p in allp:
##                        if sum(team_dict[team][5][p[0]][1])<minip:
##                            minip=sum(team_dict[team][5][p[0]][1])
##                            minp=p
##                    return(minp)
        
            else:
                if len(low_lev)>0:
                    for p in low_lev:
                        if sum(team_dict[team][5][p[0]][1])<minip:
                            minip=sum(team_dict[team][5][p[0]][1])
                            minp=p
                    return(minp)
                else:
                    return(cur_p)
        elif inn==2:
            if li>=1:
                if len(high_lev_sp)>0:
                    return(high_lev_sp[0])
                else:
                    if len(low_lev)>0:
                        for p in low_lev:
                            if sum(team_dict[team][5][p[0]][1])<minip:
                                minip=sum(team_dict[team][5][p[0]][1])
                                minp=p
                        return(minp)
                    else:
                        return(cur_p)
##            elif li>=.9:
##                if len(high_lev_sp)==2:
##                    return(high_lev_sp[1])
##                elif len(high_lev_sp)==1 and high_lev_sp[0][0]==1:
##                    return(high_lev_sp[0])
##                else:
##                    minip=100
##                    minp=-1
##                    for p in allp:
##                        if sum(team_dict[team][5][p[0]][1])<minip:
##                            minip=sum(team_dict[team][5][p[0]][1])
##                            minp=p
##                    return(minp)
        
            else:
                if len(allp)>0:
                    for p in allp:
                        if sum(team_dict[team][5][p[0]][1])<minip:
                            minip=sum(team_dict[team][5][p[0]][1])
                            minp=p
                    return(minp)
                else:
                    return(cur_p)
                

def pull_pitcher2(team_dict,team,spin,cur_p,inn,oppruns,runs,low_lev,runs_last_sub,inn_last_sub,li,high_lev_sp):
    ## conditions to pull pitcher and what pithcer to bring in
    runs_given_up=oppruns-runs_last_sub
    innp=inn-inn_last_sub
    allp=high_lev_sp+low_lev

    ### create sp, midrange, and high lev check
    lev=cur_p[1]

    
    
    
    if (pullrest(team_dict,team,cur_p,inn,runs_given_up,innp,lev,high_lev_sp,allp,li)):
##        if choose_new_p(1,team_dict,team,cur_p,inn,li,high_lev,med_lev,low_lev,high_lev_sp)==-1:
##            print('one')
##            print('-1')
##            print(high_lev)
##            print(med_lev)
##            print(low_lev)
##        elif choose_new_p(1,team_dict,team,cur_p,inn,li,high_lev,med_lev,low_lev,high_lev_sp) is None:
##            print('one')
##            print('none')
##            print(high_lev)
##            print(med_lev)
##            print(low_lev)
        return(choose_new_p(1,team_dict,team,cur_p,inn,li,low_lev,high_lev_sp))
    else:
##        if choose_new_p(0,team_dict,team,cur_p,inn,li,high_lev,med_lev,low_lev,high_lev_sp)==-1:
##            print('zero')
##            print('-1')
##            print(high_lev)
##            print(med_lev)
##            print(low_lev)
##        elif choose_new_p(0,team_dict,team,cur_p,inn,li,high_lev,med_lev,low_lev,high_lev_sp) is None :
##            print('zero')
##            print('none')
##            print(high_lev)
##            print(med_lev)
##            print(low_lev)
        
        
        return(choose_new_p(0,team_dict,team,cur_p,inn,li,low_lev,high_lev_sp))
  
               
                    
def pullstarter(inn,runs):
    if inn==0 and runs>6:
        return(True)
    elif inn==1 and runs>5:
        return(True)
    elif inn==2 and runs>5:
        return(True)
    elif inn==3 and runs>4:
        return(True)
    elif inn==4 and runs>3:
        return(True)
    elif inn==5 and runs>2:
        return(True)
    elif inn==6 and runs>1:
        return(True)
    elif inn==7 and runs>0:
        return(True)
    else:
        return(False)
                    
                        

### function to check if current mop up guy should be taken out
                        ## true= take out
def mopupcheck(cur_p,team_dict,team,inn,defc,runs_given_up,gameip):
    pastip=sum(team_dict[team][1][cur_p][1])
    if gameip>3:
        if runs_given_up==0:
            if pastip+gameip<=5:
                return(False)
            else:
                return(True)
        else:
            return(True)
    if pastip+gameip>6:
        if runs_given_up==0:
            return(False)
        else:
            return(True)
    if pastip+gameip>7:
        return(False)

    if (float(float(runs_given_up)/float(gameip))>1.5):
        return(True)
                                           
            
### function to pick new mop up guy
##def picknewmop(team_dict,team,mopaval,cur_p):
##    if len(mopaval)==0:
##        return(cur_p)
##    
##    minip=1000
##    new_p=0
##    for p in mopaval:
##        if sum(team_dict[team][1][p][1])<minip:
##            minip=sum(team_dict[team][1][p][1])
##            new_p=p
##    return(new_p)
    
                
    
    
                
    


    ### assign random games for all 30 teams 162 times


def randmatchup():
    teams=set(range(0,30))
    seen=[]
    games=[]
    while len(teams)>0:
        import random
        s=random.sample(teams, 2)
        teams.remove(s[0])
        teams.remove(s[1])
        games.append([s[0],s[1]])
    return(games)

def compare(team_dict):
    import random
    import numpy
    import pandas as pd
    from matplotlib import pyplot
    from scipy.stats.stats import pearsonr

    ## compare real rp li dist with sim rp li dist
    rpli=pd.read_csv('rpli.csv')
    spli=pd.read_csv('spli.csv')

    
    
 #   rpinnli=rpli['inLI']
    rpinnli=rpli['inLI']
    realera=rpli['ERA']

    ### compare relivers first
    
    rpinnli=rpli['inLI']
    realera=rpli['ERA']

    simERArp=[]
    sim2ERArp=[]
    simLIrp=[]
    sim2LIrp=[]
    

   

    for t in range(0,30):
        for p in range(5,12):
            sim2LIrp.append(float(team_dict[t][7][p][0])/float(team_dict[t][7][p][1]))
            sim2ERArp.append(float(team_dict[t][7][p][2])/float(team_dict[t][7][p][1]))
            simLIrp.append(float(team_dict[t][3][p][0])/float(team_dict[t][3][p][1]))
            simERArp.append(float(team_dict[t][3][p][2])/float(team_dict[t][3][p][1]))


    print('relievers')
    print('---------')
    print('real')
    print(pearsonr(rpinnli, realera))

    print('sim')
    
    print(pearsonr(simLIrp, simERArp))

    print('sim2')
    
    print(pearsonr(sim2LIrp, sim2ERArp))
    return
    de=[]
##    for p in range(len(spli)):
##        if numpy.isnan(spli['inLI'][p]):
##            print(p)
            #spli = numpy.delete(x, (0), axis=0)
 #   spli=numpy.delete(spli,de)
    spli=spli.drop(spli.index[[33,35,72,74]])

    
    spinnli=spli['inLI']
    sprealera=spli['ERA']

    

    simERAsp=[]
    sim2ERAsp=[]
    simLIsp=[]
    sim2LIsp=[]
    

   

    for t in range(0,30):
        for p in range(0,5):
            sim2LIsp.append(float(team_dict[t][7][p][0])/float(team_dict[t][7][p][1]))
            sim2ERAsp.append(float(team_dict[t][7][p][2])/float(team_dict[t][7][p][1]))
            simLIsp.append(float(team_dict[t][3][p][0])/float(team_dict[t][3][p][1]))
            simERAsp.append(float(team_dict[t][3][p][2])/float(team_dict[t][3][p][1]))


    print('starters')
    print('---------')
    print('real')
    print(pearsonr(spinnli, sprealera))

    print('sim')
    
    print(pearsonr(simLIsp, simERAsp))

    print('sim2')

    print(pearsonr(sim2LIsp, sim2ERAsp))



    allli=rpli.append(spli)

    allinnli=allli['inLI']
    allera=allli['ERA']

    sim2LIall=[]
    erasim2all=[]
    simLIall=[]
    erasimall=[]

    for t in range(0,30):
        for p in range(0,12):
            sim2LIall.append(float(team_dict[t][7][p][0])/float(team_dict[t][7][p][1]))
            erasim2all.append(float(team_dict[t][7][p][2])/float(team_dict[t][7][p][1]))
            simLIall.append(float(team_dict[t][3][p][0])/float(team_dict[t][3][p][1]))
            erasimall.append(float(team_dict[t][3][p][2])/float(team_dict[t][3][p][1]))


    
    print('ALL')
    print('---------')
    print('real')
    print(pearsonr(allinnli, allera))

    print('sim')
    
    print(pearsonr(simLIall, erasimall))

    print('sim2')

    print(pearsonr(sim2LIall, erasim2all))
 #   bins = numpy.linspace(-10, 10, 100)

#    pyplot.hist(rpinnli, bins, alpha=0.5, label='real')
 #   pyplot.hist(simLIrp, bins, alpha=0.5, label='sim')
#    pyplot.legend(loc='upper right')
#    pyplot.show()


                ## pick sp for the game.
                ## sp candidates
                #### all former sp who have pitched 6 or less in the last 7 games, and didnt pitch last game
                #### all former high lev p who have pitched 3 or less in the last week
                ### if no one left, mopup guy who is at 5 or less
##sp==3,4,5,8,9,10,11
def picksp(team_dict,team,sp,oppr):

    ## pitchers with 2 consectuive days off
    elgsp=[]
##    if team_dict[team][5][1][1][6]==0 and team_dict[team][5][1][1][5]==0 and team_dict[team][5][1][1][4]==0:
##        return([1,3])
    
    
    for i in sp:
        if (i[0]==0 or i[0]==1) and (team_dict[team][5][i[0]][1][6]==0 and team_dict[team][5][i[0]][1][5]==0 and team_dict[team][5][i[0]][1][4]==0 and team_dict[team][5][i[0]][1][3]==0 ):
            return(i)
        else:                  
            if team_dict[team][5][i[0]][1][6]==0 and team_dict[team][5][i[0]][1][5]<=1:
                elgsp.append(i)
    if len(elgsp)==0:
        minip=1000
        minp=-1
        for i in sp:
            if i[0]!=10 and i[0]!=11:
                if sum(team_dict[team][5][i[0]][1])<minip:
                    minip=sum(team_dict[team][5][i[0]][1])
                    minp=i
        return(minp)
        #if team_dict[team][5][1][1][6]==0 and team_dict[team][5][1][1][5]==0:
#            return([1,3])
        
        if sum(team_dict[team][5][11][1])<sum(team_dict[team][5][10][1]):
            if [11,1] in sp:
                return([11,1])
            elif [10,1] in sp:
                return([10,1])
        else:
            if [10,1] in sp:
                return([10,1])
            elif [11,1] in sp:
                return([11,1])
    
    
    minip=1000
    minr_p=-1
    for p in elgsp:
        if sum(team_dict[team][5][p[0]][1])<minip:
            minip=sum(team_dict[team][5][p[0]][1])
            minr_p=p
    return(minr_p)
        
    
            
            
    
def findli(inn,defc,homeind,home,away,home_p,away_p,team_dict,ret):
    
    
    
    if ret==1:
        
        away_defc=defc
        r=0
        if homeind==0:
            if inn==0:
                if away_defc<=-4:
                    r=.7
                elif away_defc==-3:
                    r=.8
                elif away_defc==-2:
                    r=.9
                elif away_defc==-1:
                    r=.9
                elif away_defc==0:
                    r=.9

            elif inn==1:
                if away_defc<=-4:
                    r=.8
                elif away_defc==-3:
                    r=.9
                elif away_defc==-2:
                    r=1
                elif away_defc==-1:
                    r=1
                elif away_defc==0:
                    r=.9
                elif away_defc==1:
                    r=.8
                elif away_defc==2:
                    r=.6
                elif away_defc==3:
                    r=.5
                elif away_defc>=4:
                    r=.4



            elif inn==2:
                if away_defc<=-4:
                    r=.8
                elif away_defc==-3:
                    r=.9
                elif away_defc==-2:
                    r=1
                elif away_defc==-1:
                    r=1.1
                elif away_defc==0:
                    r=1
                elif away_defc==1:
                    r=.8
                elif away_defc==2:
                    r=.6
                elif away_defc==3:
                    r=.5
                elif away_defc>=4:
                    r=.3



            elif inn==3:
                if away_defc<=-4:
                    r=.8
                elif away_defc==-3:
                    r=1
                elif away_defc==-2:
                    r=1.1
                elif away_defc==-1:
                    r=1.2
                elif away_defc==0:
                    r=1.1
                elif away_defc==1:
                    r=.9
                elif away_defc==2:
                    r=.6
                elif away_defc==3:
                    r=.4
                elif away_defc>=4:
                    r=.3



            elif inn==4:
                if away_defc<=-4:
                    r=.8
                elif away_defc==-3:
                    r=1.1
                elif away_defc==-2:
                    r=1.3
                elif away_defc==-1:
                    r=1.3
                elif away_defc==0:
                    r=1.2
                elif away_defc==1:
                    r=.9
                elif away_defc==2:
                    r=.6
                elif away_defc==3:
                    r=.4
                elif away_defc>=4:
                    r=.3



            elif inn==5:
                if away_defc<=-4:
                    r=.8
                elif away_defc==-3:
                    r=1.1
                elif away_defc==-2:
                    r=1.4
                elif away_defc==-1:
                    r=1.6
                elif away_defc==0:
                    r=1.3
                elif away_defc==1:
                    r=.9
                elif away_defc==2:
                    r=.6
                elif away_defc==3:
                    r=.3
                elif away_defc>=4:
                    r=.2

        


            elif inn==6:
                if away_defc<=-4:
                    r=.8
                elif away_defc==-3:
                    r=1.2
                elif away_defc==-2:
                    r=1.6
                elif away_defc==-1:
                    r=1.9
                elif away_defc==0:
                    r=1.5
                elif away_defc==1:
                    r=.8
                elif away_defc==2:
                    r=.4
                elif away_defc==3:
                    r=.2
                elif away_defc>=4:
                    r=.1



    #
            elif inn==7:
                if away_defc<=-4:
                    r=.7
                elif away_defc==-3:
                    r=1.1
                elif away_defc==-2:
                    r=1.8
                elif away_defc==-1:
                    r=2.5
                elif away_defc==0:
                    r=1.8
                elif away_defc==1:
                    r=.6
                elif away_defc==2:
                    r=.3
                elif away_defc==3:
                    r=.1
                elif away_defc>=4:
                    r=.1

           

            elif inn==8:
                if away_defc<=-4:
                    r=.5
                elif away_defc==-3:
                    r=1
                elif away_defc==-2:
                    r=2
                elif away_defc==-1:
                    r=3.6
                elif away_defc==0:
                    r=2.3
                elif away_defc==1:
                    r=.8
                elif away_defc==2:
                    r=.4
                elif away_defc==3:
                    r=.2
                elif away_defc>=4:
                    r=.1
            return(r)
        else:
            home_defc=defc
            if inn==0:
                r=.9

            elif inn==1:
                if home_defc<=-4:
                    r=.4
                elif home_defc==-3:
                    r=.6
                elif home_defc==-2:
                    r=.7
                elif home_defc==-1:
                    r=.8
                elif home_defc==0:
                    r=.9
                elif home_defc==1:
                    r=1
                elif home_defc==2:
                    r=.9
                elif home_defc==3:
                    r=.8
                elif home_defc>=4:
                    r=.7
               

            elif inn==2:
                if home_defc<=-4:
                    r=.4
                elif home_defc==-3:
                    r=.6
                elif home_defc==-2:
                    r=.7
                elif home_defc==-1:
                    r=.9
                elif home_defc==0:
                    r=1
                elif home_defc==1:
                    r=1
                elif home_defc==2:
                    r=1
                elif home_defc==3:
                    r=.9
                elif home_defc>=4:
                    r=.7
              

            elif inn==3:
                if home_defc<=-4:
                    r=.4
                elif home_defc==-3:
                    r=.5
                elif home_defc==-2:
                    r=.7
                elif home_defc==-1:
                    r=.9
                elif home_defc==0:
                    r=1.1
                elif home_defc==1:
                    r=1.1
                elif home_defc==2:
                    r=1.1
                elif home_defc==3:
                    r=.9
                elif home_defc>=4:
                    r=.7

            elif inn==4:
                if home_defc<=-4:
                    r=.4
                elif home_defc==-3:
                    r=.5
                elif home_defc==-2:
                    r=.7
                elif home_defc==-1:
                    r=1
                elif home_defc==0:
                    r=1.2
                elif home_defc==1:
                    r=1.3
                elif home_defc==2:
                    r=1.1
                elif home_defc==3:
                    r=.9
                elif home_defc>=4:
                    r=.7


            elif inn==5:
                if home_defc<=-4:
                    r=.3
                elif home_defc==-3:
                    r=.5
                elif home_defc==-2:
                    r=.7
                elif home_defc==-1:
                    r=1
                elif home_defc==0:
                    r=1.3
                elif home_defc==1:
                    r=1.4
                elif home_defc==2:
                    r=1.3
                elif home_defc==3:
                    r=1
                elif home_defc>=4:
                    r=.7



            elif inn==6:
                if home_defc<=-4:
                    r=.2
                elif home_defc==-3:
                    r=.4
                elif home_defc==-2:
                    r=.7
                elif home_defc==-1:
                    r=1
                elif home_defc==0:
                    r=1.5
                elif home_defc==1:
                    r=1.7
                elif home_defc==2:
                    r=1.4
                elif home_defc==3:
                    r=1
                elif home_defc>=4:
                    r=.6



            elif inn==7:
                if home_defc<=-4:
                    r=.2
                elif home_defc==-3:
                    r=.3
                elif home_defc==-2:
                    r=.6
                elif home_defc==-1:
                    r=1
                elif home_defc==0:
                    r=1.9
                elif home_defc==1:
                    r=2.2
                elif home_defc==2:
                    r=1.5
                elif home_defc==3:
                    r=.9
                elif home_defc>=4:
                    r=.6



            elif inn==8:
                if home_defc<=-4:
                    r=.1
                elif home_defc==-3:
                    r=.2
                elif home_defc==-2:
                    r=.3
                elif home_defc==-1:
                    r=.7
                elif home_defc==0:
                    r=2.4
                elif home_defc==1:
                    r=2.9
                elif home_defc==2:
                    r=1.6
                elif home_defc==3:
                    r=.8
                elif home_defc>=4:
                    r=.4
            return(r)
    else:
        if away_p is None:
            print('NONE')
            
        if home_p is None:
            print('NONE')
      
        away_defc=defc
        if homeind==0:
            if inn==0:
                if away_defc<=-4:
                    team_dict[away][3][away_p][0]+=.7
                elif away_defc==-3:
                    team_dict[away][3][away_p][0]+=.8
                elif away_defc==-2:
                    team_dict[away][3][away_p][0]+=.9
                elif away_defc==-1:
                    team_dict[away][3][away_p][0]+=.9
                elif away_defc==0:
                    team_dict[away][3][away_p][0]+=.9


                

            elif inn==1:
                if away_defc<=-4:
                    team_dict[away][3][away_p][0]+=.8
                elif away_defc==-3:
                    team_dict[away][3][away_p][0]+=.9
                elif away_defc==-2:
                    team_dict[away][3][away_p][0]+=1
                elif away_defc==-1:
                    team_dict[away][3][away_p][0]+=1
                elif away_defc==0:
                    team_dict[away][3][away_p][0]+=.9
                elif away_defc==1:
                    team_dict[away][3][away_p][0]+=.8
                elif away_defc==2:
                    team_dict[away][3][away_p][0]+=.6
                elif away_defc==3:
                    team_dict[away][3][away_p][0]+=.5
                elif away_defc>=4:
                    team_dict[away][3][away_p][0]+=.4



            elif inn==2:
                if away_defc<=-4:
                    team_dict[away][3][away_p][0]+=.8
                elif away_defc==-3:
                    team_dict[away][3][away_p][0]+=.9
                elif away_defc==-2:
                    team_dict[away][3][away_p][0]+=1
                elif away_defc==-1:
                    team_dict[away][3][away_p][0]+=1.1
                elif away_defc==0:
                    team_dict[away][3][away_p][0]+=1
                elif away_defc==1:
                    team_dict[away][3][away_p][0]+=.8
                elif away_defc==2:
                    team_dict[away][3][away_p][0]+=.6
                elif away_defc==3:
                    team_dict[away][3][away_p][0]+=.5
                elif away_defc>=4:
                    team_dict[away][3][away_p][0]+=.3



            elif inn==3:
                if away_defc<=-4:
                    team_dict[away][3][away_p][0]+=.8
                elif away_defc==-3:
                    team_dict[away][3][away_p][0]+=1
                elif away_defc==-2:
                    team_dict[away][3][away_p][0]+=1.1
                elif away_defc==-1:
                    team_dict[away][3][away_p][0]+=1.2
                elif away_defc==0:
                    team_dict[away][3][away_p][0]+=1.1
                elif away_defc==1:
                    team_dict[away][3][away_p][0]+=.9
                elif away_defc==2:
                    team_dict[away][3][away_p][0]+=.6
                elif away_defc==3:
                    team_dict[away][3][away_p][0]+=.4
                elif away_defc>=4:
                    team_dict[away][3][away_p][0]+=.3



            elif inn==4:
                if away_defc<=-4:
                    team_dict[away][3][away_p][0]+=.8
                elif away_defc==-3:
                    team_dict[away][3][away_p][0]+=1.1
                elif away_defc==-2:
                    team_dict[away][3][away_p][0]+=1.3
                elif away_defc==-1:
                    team_dict[away][3][away_p][0]+=1.3
                elif away_defc==0:
                    team_dict[away][3][away_p][0]+=1.2
                elif away_defc==1:
                    team_dict[away][3][away_p][0]+=.9
                elif away_defc==2:
                    team_dict[away][3][away_p][0]+=.6
                elif away_defc==3:
                    team_dict[away][3][away_p][0]+=.4
                elif away_defc>=4:
                    team_dict[away][3][away_p][0]+=.3



            elif inn==5:
                if away_defc<=-4:
                    team_dict[away][3][away_p][0]+=.8
                elif away_defc==-3:
                    team_dict[away][3][away_p][0]+=1.1
                elif away_defc==-2:
                    team_dict[away][3][away_p][0]+=1.4
                elif away_defc==-1:
                    team_dict[away][3][away_p][0]+=1.6
                elif away_defc==0:
                    team_dict[away][3][away_p][0]+=1.3
                elif away_defc==1:
                    team_dict[away][3][away_p][0]+=.9
                elif away_defc==2:
                    team_dict[away][3][away_p][0]+=.6
                elif away_defc==3:
                    team_dict[away][3][away_p][0]+=.3
                elif away_defc>=4:
                    team_dict[away][3][away_p][0]+=.2

        


            elif inn==6:
                if away_defc<=-4:
                    team_dict[away][3][away_p][0]+=.8
                elif away_defc==-3:
                    team_dict[away][3][away_p][0]+=1.2
                elif away_defc==-2:
                    team_dict[away][3][away_p][0]+=1.6
                elif away_defc==-1:
                    team_dict[away][3][away_p][0]+=1.9
                elif away_defc==0:
                    team_dict[away][3][away_p][0]+=1.5
                elif away_defc==1:
                    team_dict[away][3][away_p][0]+=.8
                elif away_defc==2:
                    team_dict[away][3][away_p][0]+=.4
                elif away_defc==3:
                    team_dict[away][3][away_p][0]+=.2
                elif away_defc>=4:
                    team_dict[away][3][away_p][0]+=.1



    #
            elif inn==7:
                if away_defc<=-4:
                    team_dict[away][3][away_p][0]+=.7
                elif away_defc==-3:
                    team_dict[away][3][away_p][0]+=1.1
                elif away_defc==-2:
                    team_dict[away][3][away_p][0]+=1.8
                elif away_defc==-1:
                    team_dict[away][3][away_p][0]+=2.5
                elif away_defc==0:
                    team_dict[away][3][away_p][0]+=1.8
                elif away_defc==1:
                    team_dict[away][3][away_p][0]+=.6
                elif away_defc==2:
                    team_dict[away][3][away_p][0]+=.3
                elif away_defc==3:
                    team_dict[away][3][away_p][0]+=.1
                elif away_defc>=4:
                    team_dict[away][3][away_p][0]+=.1

           

            elif inn==8:
                if away_defc<=-4:
                    team_dict[away][3][away_p][0]+=.5
                elif away_defc==-3:
                    team_dict[away][3][away_p][0]+=1
                elif away_defc==-2:
                    team_dict[away][3][away_p][0]+=2
                elif away_defc==-1:
                    team_dict[away][3][away_p][0]+=3.6
                elif away_defc==0:
                    team_dict[away][3][away_p][0]+=2.3
                elif away_defc==1:
                    team_dict[away][3][away_p][0]+=.8
                elif away_defc==2:
                    team_dict[away][3][away_p][0]+=.4
                elif away_defc==3:
                    team_dict[away][3][away_p][0]+=.2
                elif away_defc>=4:
                    team_dict[away][3][away_p][0]+=.1
        else:
    
            home_defc=defc
            if inn==0:
                team_dict[home][3][home_p][0]+=.9

            elif inn==1:
                if home_defc<=-4:
                    team_dict[home][3][home_p][0]+=.4
                elif home_defc==-3:
                    team_dict[home][3][home_p][0]+=.6
                elif home_defc==-2:
                    team_dict[home][3][home_p][0]+=.7
                elif home_defc==-1:
                    team_dict[home][3][home_p][0]+=.8
                elif home_defc==0:
                    team_dict[home][3][home_p][0]+=.9
                elif home_defc==1:
                    team_dict[home][3][home_p][0]+=1
                elif home_defc==2:
                    team_dict[home][3][home_p][0]+=.9
                elif home_defc==3:
                    team_dict[home][3][home_p][0]+=.8
                elif home_defc>=4:
                    team_dict[home][3][home_p][0]+=.7
               

            elif inn==2:
                if home_defc<=-4:
                    team_dict[home][3][home_p][0]+=.4
                elif home_defc==-3:
                    team_dict[home][3][home_p][0]+=.6
                elif home_defc==-2:
                    team_dict[home][3][home_p][0]+=.7
                elif home_defc==-1:
                    team_dict[home][3][home_p][0]+=.9
                elif home_defc==0:
                    team_dict[home][3][home_p][0]+=1
                elif home_defc==1:
                    team_dict[home][3][home_p][0]+=1
                elif home_defc==2:
                    team_dict[home][3][home_p][0]+=1
                elif home_defc==3:
                    team_dict[home][3][home_p][0]+=.9
                elif home_defc>=4:
                    team_dict[home][3][home_p][0]+=.7
              

            elif inn==3:
                if home_defc<=-4:
                    team_dict[home][3][home_p][0]+=.4
                elif home_defc==-3:
                    team_dict[home][3][home_p][0]+=.5
                elif home_defc==-2:
                    team_dict[home][3][home_p][0]+=.7
                elif home_defc==-1:
                    team_dict[home][3][home_p][0]+=.9
                elif home_defc==0:
                    team_dict[home][3][home_p][0]+=1.1
                elif home_defc==1:
                    team_dict[home][3][home_p][0]+=1.1
                elif home_defc==2:
                    team_dict[home][3][home_p][0]+=1.1
                elif home_defc==3:
                    team_dict[home][3][home_p][0]+=.9
                elif home_defc>=4:
                    team_dict[home][3][home_p][0]+=.7

            elif inn==4:
                if home_defc<=-4:
                    team_dict[home][3][home_p][0]+=.4
                elif home_defc==-3:
                    team_dict[home][3][home_p][0]+=.5
                elif home_defc==-2:
                    team_dict[home][3][home_p][0]+=.7
                elif home_defc==-1:
                    team_dict[home][3][home_p][0]+=1
                elif home_defc==0:
                    team_dict[home][3][home_p][0]+=1.2
                elif home_defc==1:
                    team_dict[home][3][home_p][0]+=1.3
                elif home_defc==2:
                    team_dict[home][3][home_p][0]+=1.1
                elif home_defc==3:
                    team_dict[home][3][home_p][0]+=.9
                elif home_defc>=4:
                    team_dict[home][3][home_p][0]+=.7


            elif inn==5:
                if home_defc<=-4:
                    team_dict[home][3][home_p][0]+=.3
                elif home_defc==-3:
                    team_dict[home][3][home_p][0]+=.5
                elif home_defc==-2:
                    team_dict[home][3][home_p][0]+=.7
                elif home_defc==-1:
                    team_dict[home][3][home_p][0]+=1
                elif home_defc==0:
                    team_dict[home][3][home_p][0]+=1.3
                elif home_defc==1:
                    team_dict[home][3][home_p][0]+=1.4
                elif home_defc==2:
                    team_dict[home][3][home_p][0]+=1.3
                elif home_defc==3:
                    team_dict[home][3][home_p][0]+=1
                elif home_defc>=4:
                    team_dict[home][3][home_p][0]+=.7



            elif inn==6:
                if home_defc<=-4:
                    team_dict[home][3][home_p][0]+=.2
                elif home_defc==-3:
                    team_dict[home][3][home_p][0]+=.4
                elif home_defc==-2:
                    team_dict[home][3][home_p][0]+=.7
                elif home_defc==-1:
                    team_dict[home][3][home_p][0]+=1
                elif home_defc==0:
                    team_dict[home][3][home_p][0]+=1.5
                elif home_defc==1:
                    team_dict[home][3][home_p][0]+=1.7
                elif home_defc==2:
                    team_dict[home][3][home_p][0]+=1.4
                elif home_defc==3:
                    team_dict[home][3][home_p][0]+=1
                elif home_defc>=4:
                    team_dict[home][3][home_p][0]+=.6



            elif inn==7:
                if home_defc<=-4:
                    team_dict[home][3][home_p][0]+=.2
                elif home_defc==-3:
                    team_dict[home][3][home_p][0]+=.3
                elif home_defc==-2:
                    team_dict[home][3][home_p][0]+=.6
                elif home_defc==-1:
                    team_dict[home][3][home_p][0]+=1
                elif home_defc==0:
                    team_dict[home][3][home_p][0]+=1.9
                elif home_defc==1:
                    team_dict[home][3][home_p][0]+=2.2
                elif home_defc==2:
                    team_dict[home][3][home_p][0]+=1.5
                elif home_defc==3:
                    team_dict[home][3][home_p][0]+=.9
                elif home_defc>=4:
                    team_dict[home][3][home_p][0]+=.6



            elif inn==8:
                if home_defc<=-4:
                    team_dict[home][3][home_p][0]+=.1
                elif home_defc==-3:
                    team_dict[home][3][home_p][0]+=.2
                elif home_defc==-2:
                    team_dict[home][3][home_p][0]+=.3
                elif home_defc==-1:
                    team_dict[home][3][home_p][0]+=.7
                elif home_defc==0:
                    team_dict[home][3][home_p][0]+=2.4
                elif home_defc==1:
                    team_dict[home][3][home_p][0]+=2.9
                elif home_defc==2:
                    team_dict[home][3][home_p][0]+=1.6
                elif home_defc==3:
                    team_dict[home][3][home_p][0]+=.8
                elif home_defc>=4:
                    team_dict[home][3][home_p][0]+=.4
    





    










