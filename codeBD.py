# import cherrypy
import mysql.connector
# import os
import connectionDB


class requettesBD(object):
    def selectBD(self,lig):
        if(lig!=-1):
            sql = "SELECT * FROM echantillonGeneralIndividuelle WHERE id_echantillonGeneral= %s"
            connectionDB.cursor.execute(sql,(lig,))

                # Récupération des résultats
            results = connectionDB.cursor.fetchall()
                # Création des tableaux
            tableau = []
                # Parcours des résultats
            for row in results:
                    # Traitement des données et ajout dans le tableau
                ligne = [row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10], row[11], row[12], row[13], row[14], row[15], row[16], row[17], row[18]]
                tableau.append(ligne)
            return tableau
        else:
            connectionDB.cursor.close()
            connectionDB.cnx.close()
            return []
    
    def secondselectBD(self,targ):
        if(targ!=-1):
            sql = "SELECT * FROM echantillonGeneralIndividuelle WHERE formation= %s"
            connectionDB.cursor.execute(sql,(targ,))

                # Récupération des résultats
            results = connectionDB.cursor.fetchall()
                # Création des tableaux
            tableau = []
                # Parcours des résultats
            for row in results:
                    # Traitement des données et ajout dans le tableau
                ligne = [row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10], row[11], row[12], row[13], row[14], row[15], row[16], row[17], row[18]]
                tableau.append(ligne)
            return tableau
        else:
            connectionDB.cursor.close()
            connectionDB.cnx.close()
            return []
        
# if __name__ == '__main__':
    # cherrypy.quickstart(FormHandler(),config ="configuration.conf")
