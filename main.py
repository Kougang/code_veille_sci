import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from codeBD import requettesBD
import os
import cherrypy
from numpy.core.numeric import True_


# ===========================================================================================
# ===========================================================================================
# =============ATTENTION--ATTENTION==========================================================
# ===========================================================================================
# ===========================================================================================

# cherchez et lisez le fichier lisMoi.txt pour comprendre les conventions et le mode de traitement  et fonctionnnement de ce code


class MainComponent(object):
    def fluctuation (self):
        with open('dossierBase/index.html','r') as f:
            flux= f.read()
        return flux 
    fluctuation.exposed=True
    
    def tesModel(self,nameFile,nbrVoisin,skipr=0,features_test=[[]],target_test=[]): 
        # print("bien recu")
        recommendation = np.loadtxt(nameFile,delimiter=',', skiprows=skipr)
        X_train = recommendation[:, :-1]
        y_train =recommendation[:, -1]
        
            # Create an instance of the classifier
            # self.nVoisin=nbrVoisin
        knn = KNeighborsClassifier(n_neighbors=nbrVoisin)
            # Train the classifier on the data
        knn.fit(X_train, y_train)
        
        X_test=np.array(features_test)
        y_test=np.array(target_test)
        
        y_pred = knn.predict(X_test)
            # determiner la precision 
        print(y_pred)
        accuracy = knn.score(X_test, y_test)
        
        prec="Précision de la prédiction : {:.2f}%".format(accuracy * 100)
        
        with open('./dossierBase/main.html','r') as f:
            flux= f.read()
        
        return prec
        
    def index(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        conf = {
            '/': {
                'tools.sessions.on': True,
                'tools.staticdir.root': current_dir
            },
            '/static': {
                'tools.staticdir.on': True,
                'tools.staticdir.dir': 'dossierBase'
            },
            '/css': {
                'tools.staticdir.on': True,
                'tools.staticdir.dir': 'dossierBase/css'
            }
        }
        cherrypy.tree.mount(MainComponent(), '/', conf)
        with open('dossierBase/index.html') as f:
            return f.read()
    index.exposed=True
    
    def train_data(self,nameFile='echantillonGeneral.csv',skipr=1,nbrVoisin=5,math=0, physique=0, info=0, chimie=0, philo=0, svt=0, hist=0, geo=0, ecm=0, ang=0, sport=0):
        
        
        math    = int(math)
        physique= int(physique)
        info    = int(info)
        chimie  = int(chimie)
        philo   = int(philo)
        svt     = int(svt)
        hist    = int(hist)
        geo     = int(geo)
        ecm     = int(ecm)
        ang     = int(ang)
        sport   = int(sport)

        # features=int(features)
        recommendation = np.loadtxt(nameFile,delimiter=',', skiprows=skipr)
        X_train = recommendation[:, :-1]
        y_train =recommendation[:, -1]
        
            # Create an instance of the classifier
        knn = KNeighborsClassifier(n_neighbors=nbrVoisin)
            # Train the classifier on the data
        knn.fit(X_train, y_train)
        
        X_test=np.array([[math, physique, info, chimie, philo, svt, hist, geo, ecm, ang, sport]])

        distances, indices = knn.kneighbors(X_test,nbrVoisin)

        # optenir la clsasse des plus proches voisins trouvés
        neighbor_classes = y_train[indices]
        
        # applanissement de la table d indice
        ind=[]
        for rows in indices:
            for r in rows:
                ind.append(r)
        # applanissement de la table des targets
        targ=[]
        for lig in neighbor_classes:
            for elt in lig:
                targ.append(elt)
                
# ===========================================================================================================
        # MainComponent().K_EchantIndividuel(ind)
        # MainComponent().trait_target(ind,targ)
# ===========================================================================================================

        indice=ind
        target = targ
                
        # SUPPRIMONS TOUS LES ELEMENTS QUI SE REPETENT DANS NOTRE TARGET
        # noRepeatTarget=list(set(target))
        noRepeatTarget=[]
        i=0
        inter=[]
        for num in target:
            if num not in noRepeatTarget:
                noRepeatTarget.append(num)
                inter.append(i)
                i+=1
            else:
                i+=1
        # print("indice intermediauire",inter)
        
            # retenons uniquement les indices des elements qui ont ete retenu dans les targets
        
        indiceFinaux=[]
        for elt in inter:
            indiceFinaux.append(indice[elt])
            
        # print("indice finale",indiceFinale)  
        
            # conversion des valeurs des targets obtenues en entier  
        convertInt=[]
        for first in noRepeatTarget:
            convertInt.append(int(first))
        
        
        # return "voici le bloc trait_target"
# ===========================================================================================================

        # MainComponent().appelBD(indiceFinaux,convertInt)
# ===========================================================================================================

        convertInt.append(-1)
        tmp=[]
        # ici on evite de faire la selection basé sur une meme classe plusieurs fois
        
        firstTarget=[]
        secondTarget=[]
        thirdTarget=[]
        fourTarget=[]
        closedb=[]
        
        print("les targets",convertInt)
        
        for tar in convertInt:
            if tar not in tmp:
                if(tar==0):
                    firstTarget=requettesBD().secondselectBD(tar)
                    tmp.append(tar)
                if(tar==1):
                    secondTarget=requettesBD().secondselectBD(tar)
                    tmp.append(tar)
                if(tar==2):
                    thirdTarget=requettesBD().secondselectBD(tar)
                    tmp.append(tar)    
                if(tar==3):
                    fourTarget=requettesBD().secondselectBD(tar)
                    tmp.append(tar)
# ===========================================================================================================

        # MainComponent().pre_traitement_profond(firstTarget,secondTarget,thirdTarget,fourTarget) 
# ===========================================================================================================

        



# ===============================================
        # # flux= MainComponent().index()
        mes1=0
        mes2=0
        mes3=0
        mes4=0
        var1=-1
        if(firstTarget!=[]):
        
        
            inscrit1=0
            valides1=0
            reuissit_sociale1=0
            taux_reuisit = 0
            nbr_echec1=0
            flag1=-1
            elt1=-1
            elt2=-2
            zero=0
            un=0
            i=0
            for vecteur in firstTarget:
                inscrit1=inscrit1+vecteur[15]
                valides1=valides1+vecteur[16]
                # la position [17]contient l information sur la reussit social
                if(vecteur[17]==0):
                    zero=zero+1
                else:
                    un=un+1
                i=i+1    
                reuissit_sociale1=reuissit_sociale1+vecteur[17]
            nbr_echec1=inscrit1-valides1
            taux_reuisit=valides1/inscrit1
            
            if(nbr_echec1>=(inscrit1/2)):
                flag1=0
            else:
                flag1=1

            if((zero>(i/2)) and flag1==0):
                elt1=0
                elt2=taux_reuisit
                
            if((un>=(i/2)) and flag1==1):
                elt1=0
                elt2=taux_reuisit  
                
            elt22=float(round(taux_reuisit,2))
            elt222 =int(elt22*100)
            elt2="{:.2f}%".format(elt222)

            
            
            flux=MainComponent().fluctuation()
            mes1="vous avez",elt2,"de vous integrer en MATHEMATIQUE "
            var1="vous avez",elt2,"de vous integrer en MATHEMATIQUE "
            var=0
                
            # if(var==0):    
            #     return flux.format(var1=0,mes=mes1)
            # else:
            #     return flux.format(var1=var1)       

            
        
                
        
# =============++++++++++++++++++==========================++++++++++++++++++=============
        if(secondTarget!=[]):
            inscrit1=0
            valides1=0
            reuissit_sociale1=0
            taux_reuisit = 0
            nbr_echec1=0
            flag1=-1
            elt1=-1
            elt2=-2
            zero=0
            un=0
            i=0
            for vecteur in secondTarget:
                inscrit1=inscrit1+vecteur[15]
                valides1=valides1+vecteur[16]
                # la position [17]contient l information sur la reussit social
                if(vecteur[17]==0):
                    zero=zero+1
                else:
                    un=un+1
                i=i+1    
                reuissit_sociale1=reuissit_sociale1+vecteur[17]
            nbr_echec1=inscrit1-valides1
            taux_reuisit=valides1/inscrit1
            
            if(nbr_echec1>=(inscrit1/2)):
                flag1=0
            else:
                flag1=1

            if((zero>(i/2)) and flag1==0):
                elt1=0
                elt2=taux_reuisit
                
            if((un>=(i/2)) and flag1==1):
                elt1=0
                elt2=taux_reuisit 
                
            elt22=float(round(taux_reuisit,2))
            elt222 =int(elt22*100)
            elt2="{:.2f}%".format(elt222)
                
            flux=MainComponent().fluctuation()
            mes2="vous avez",elt2,"de vous integrer en PHYSIQUE  "
            var1="vous avez",elt2,"de vous integrer en PHYSIQUE  "
            var=1
                
            # if(var==1):    
            #     return flux.format(var1=0,mes=mes2)
            # else:
            #     return flux.format(var1=var1)
    
                
# =============++++++++++++++++++==========================++++++++++++++++++=============

        if(thirdTarget!=[]):
            inscrit1=0
            valides1=0
            reuissit_sociale1=0
            taux_reuisit = 0
            nbr_echec1=0
            flag1=-1
            elt1=-1
            elt2=-2
            zero=0
            un=0
            i=0
            for vecteur in thirdTarget:
                inscrit1=inscrit1+vecteur[15]
                valides1=valides1+vecteur[16]
                # la position [17]contient l information sur la reussit social
                if(vecteur[17]==0):
                    zero=zero+1
                else:
                    un=un+1
                i=i+1    
                reuissit_sociale1=reuissit_sociale1+vecteur[17]
            nbr_echec1=inscrit1-valides1
            taux_reuisit=valides1/inscrit1
            
            if(nbr_echec1>=(inscrit1/2)):
                flag1=0
            else:
                flag1=1

            if((zero>(i/2)) and flag1==0):
                elt1=0
                elt2=taux_reuisit
                
            if((un>=(i/2)) and flag1==1):
                elt1=0
                elt2=taux_reuisit 
            elt22=float(round(taux_reuisit,2))
            elt222 =int(elt22*100)
            elt2="{:.2f}%".format(elt222)
                
            flux=MainComponent().fluctuation()
            mes3="vous avez ",elt2," de vous integrer en INFORMATIQUE"
            var=2
            var1="vous avez ",elt2," de vous integrer en INFORMATIQUE"     
            # if(var==2):    
            #     return flux.format(var1=0,mes=mes3)
            # else:
            #     return flux.format(var1=var1) 
            
# =============++++++++++++++++++==========================++++++++++++++++++=============

        if(fourTarget!=[]):
            inscrit1=0
            valides1=0
            reuissit_sociale1=0
            taux_reuisit = 0
            nbr_echec1=0
            flag1=-1
            elt1=-1
            elt2=-2
            zero=0
            un=0
            i=0
            for vecteur in fourTarget:
                inscrit1=inscrit1+vecteur[15]
                valides1=valides1+vecteur[16]
                # la position [17]contient l information sur la reussit social
                if(vecteur[17]==0):
                    zero=zero+1
                else:
                    un=un+1
                i=i+1    
                reuissit_sociale1=reuissit_sociale1+vecteur[17]
            nbr_echec1=inscrit1-valides1
            taux_reuisit=valides1/inscrit1
            
            if(nbr_echec1>=(inscrit1/2)):
                flag1=0
            else:
                flag1=1

            if((zero>(i/2)) and flag1==0):
                elt1=0
                elt2=taux_reuisit
                
            if((un>=(i/2)) and flag1==1):
                elt1=0
                elt2=taux_reuisit 
            elt22=float(round(taux_reuisit,2))
            elt222 =int(elt22*100)
            elt2="{:.2f}%".format(elt222)
                
            flux=MainComponent().fluctuation()
            mes4="vous avez ",elt2," de vous integrez en CHIMIE"
            var="<p>{mes4}</p>"
            
            var1="vous avez ",elt2," de vous integrez en CHIMIE"
            
            # if(var==3):    
            #     return flux.format(var1=0,mes=mes4)
            # else:
            #     return flux.format(var1=var1)
            
        
        prec=MainComponent().tesModel('echantillonGeneral.csv',8,1,features_test,target_test)
        flux=MainComponent().fluctuation()

        
        
        if(mes1!=0 or mes3!=0):
            Freponse=mes1
            Rreponse=mes3
            if(mes1!=0 and mes3!=0):
                with open('dossierBase/index.html','r') as f:
                    flux= f.read()
                return flux.format(Freponse=mes1,mes9=mes3,precision=prec)
            else:
                if(mes1!=0):
                    mes3=0
                    with open('dossierBase/index.html','r') as f:
                        flux= f.read()
                    return flux.format(Freponse=mes1,Rreponse=mes3,precision=prec)
                else:
                    mes1=0
                    with open('dossierBase/index.html','r') as f:
                        flux= f.read()
                    return flux.format(Freponse=mes1,Rreponse=mes3,precision=prec)
                    
        if(mes2!=0 or mes4!=0):
        
            if(mes2!=0 and mes4!=0):
                with open('dossierBase/index.html','r') as f:
                    flux= f.read()
                return flux.format(Freponse=mes2,Rreponse=mes4,precision=prec)
            else:
                if(mes2!=0):
                    mes4=0
                    with open('dossierBase/index.html','r') as f:
                        flux= f.read()
                    return flux.format(Freponse=mes2,Rreponse=mes4,precision=prec)
                else:
                    mes2=0
                    with open('dossierBase/index.html','r') as f:
                        flux= f.read()
                    return flux.format(Freponse=mes2,Rreponse=mes4,precision=prec)
        
        
        
    train_data.exposed=True


        
            
            
    


    
    
    
    
if __name__ == '__main__':
    
    features_test=np.array([
        [19,18,18,17,14,17,16,14,13,15,16],
        [19,18,18,17,16,17,15,15,16,14,15],
        [18,17,17,16,15,16,16,15,16,16,19],
        [19,17,18,18,15,16,14,16,16,16,17],
        [19,18,18,18,17,16,15,14,18,17,13],
        [20,18,17,17,16,16,16,15,17,17,16],
        [20,18,17,17,16,16,16,15,17,17,17],       
        [17,19,16,16,11,11,11,10,11,12,11],
        [17,19,17,16,11,11,11,10,11,12,11],
        [17,18,17,17,11,12,11,10,11,12,11],
        [17,18,18,17,11,12,11,10,11,12,11],
        [17,18,18,17,12,12,11,10,11,12,11],
        [17,18,18,17,12,12,11,10,11,12,12],
        [17,18,18,17,12,12,11,11,11,12,12],
        [18,17,19,15,14,13,15,14,15,15,19],
        [18,17,19,15,14,14,15,14,15,15,19],
        [17,18,19,18,15,16,13,14,15,15,16],
        [18,18,19,18,15,16,13,14,15,15,16],
        [18,18,19,18,16,16,13,14,15,15,16],
        [18,18,19,18,16,16,14,14,15,15,16],
        [17,18,19,18,16,17,15,14,15,15,16],
        [14,16,14,14,11,11,11,10,11,12,11],
        [14,16,14,14,12,11,11,10,11,12,11],
        [14,16,14,14,12,12,11,10,11,12,11],
        [14,16,14,14,12,12,11,11,11,12,11],
        [15,17,15,15,11,11,11,10,11,12,11],
        [15,17,15,15,11,11,11,11,11,12,11],
        [15,17,15,15,11,12,11,11,11,12,11],
        [15,17,15,15,11,12,11,11,12,12,11],
        [16,18,16,16,11,11,11,10,11,12,11]
    ])
    target_test=np.array([0,0,0,0,0,0,0,1,1,1,1,1,1,1,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3])
    
    
    # MainComponent().tesModel('echantillonGeneral.csv',8,1,features_test,target_test)

    # MainComponent().train_data()
    cherrypy.quickstart(MainComponent(),config ="configuration.conf")



    

