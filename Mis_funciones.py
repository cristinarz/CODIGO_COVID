import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import scipy.stats as stats
import operator
import pylab as pl
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns

    
###############################################################################
# Método para  graficar  "columna" de los 32 estados.                         #
# Muestra dos graficas la primera muestra el porcentaje de "columna" en UCI=1 #
# La segunda gráfica muestra el porcentaje de UCI en "columna"                #
###############################################################################     
def Columnas_SI_NO(df_covid,columna,color1,color2,color3,color4):  
        
    df_enf=df_covid[(df_covid[columna]<3)] #solo tomar en cuenta valores 1 y 2 de enferemedad o padecimiento
   
    si_renglon=[]
    no_renglon=[]    
    si_columna=[]
    no_columna=[]
    
    indice_estado=np.arange(1,33) #indices para estados   
    
    for i in range (32):#ciclo para recorrer los 32 estados   
         
        df_estado=df_enf[(df_enf['ENTIDAD_RES']==indice_estado[i])]  #Selecciono renglones de estado indice_estado[i]  
        #tabla de contingencia(porcentajes por renglon) de UCI y "columna"  del estado indice_estado[i] 
        tabla_contingencia_renglon=pd.crosstab(index=df_estado['UCI'], columns=df_estado[columna]).apply(lambda r: r/r.sum() *100,axis=1)
        tabla_contingencia_array_renglon=np.array(tabla_contingencia_renglon)        
        
        #porcentaje de "columna" que no estuvo en UCI
        no_renglon.append(round(tabla_contingencia_array_renglon[0][1],6))
        #porcentaje de "columna" que no estuvo en UCI
        si_renglon.append(round(tabla_contingencia_array_renglon[0][0],6))
        
        #tabla de contingencia (porcentajes por columna) de UCI y "columna" del estado indice_estado[i]
        tabla_contingencia_columna=pd.crosstab(index=df_estado['UCI'], columns=df_estado[columna]).apply(lambda r: r/r.sum() *100,axis=0)
        tabla_contingencia_array_columna=np.array(tabla_contingencia_columna)
       
        #porcentaje de "columna" que no estuvieron en "UCI"
        no_columna.append(round(tabla_contingencia_array_columna[0][1],6))
        #porcentaje de "columna" que si estuvieron en "UCI"
        si_columna.append(round(tabla_contingencia_array_columna[0][0],6))
        
    # Grafica eporcentaje por renglones   
    graficar_si_no(si_renglon,no_renglon,columna,color1,color2,indice_estado,"Porcentaje en UCI")
    
    # Grafica de porcentaje por columnas
    graficar_si_no(si_columna,no_columna,columna,color3,color4,indice_estado,"Porcentaje que estuvo em UCI de Columna")
    
##########################################################################
# Gráfica de barras de   "columna"de los 32 estados#
########################################################################## 
def graficar_si_no(si,no,columna,color1,color2,indice_estado,etiqueta_y):
    
    #Grfica porcentaje de "columna" que no estuvo en UCI
    plt.bar(np.arange(0.25,32,1), no, label = 'No', width = 0.5, color = color1)
    #Grfica  porcentaje de "columna" que  estuvo en UCI
    plt.bar(np.arange(0.75,32,1), si, label = 'Si', width = 0.5, color = color2 )
    plt.title(columna)
    plt.ylabel(etiqueta_y)
    plt.xlabel('Estados')
    #número de cada estado
    plt.xticks(indice_estado+0.38, [
            "1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24",
            "25","26","27","28","29","30","31","32"])
    plt.legend()
    plt.show()  
        
        
##########################################################################
# Método que grafica  "TIPO_PACIENTE" que estuvieron en UCI              # 
# de los 32 estados                                                      #
##########################################################################          
def Columna_tipo_paciente(df):
    
    ambulatorio=[]
    hospitalizado=[]
    
    ind=np.arange(1,33)  
    for i in range (32): 
        
      
        df=df[(df['ENTIDAD_RES']==ind[i])]  #Selecciono renglones del estado indice_estado[i]         
        ct=pd.crosstab(index=df['UCI'], columns=df['TIPO_PACIENTE']).apply(lambda r: r/r.sum() *100,axis=1)        
        ct_array=np.array(ct)        
        hospitalizado.append(round(ct_array[0][1],6))
        #ambulatorio.append(round(ct_array[0][0],6))
        
  
    plt.bar(np.arange(0.25,32,1), hospitalizado, label = 'Hospitalizado', width = 0.5, color = 'red')
    #plt.bar(np.arange(0.75,32,1) , ambulatorio , label = 'ambulatorio', width = 0.5, color = 'black')
    plt.title('TIPO_PACIENTE DE CADA ESTADO')
    plt.ylabel('% TIPO_PACIENTE en UCI ')
    plt.xlabel('estados')
    #plt.xticks(X+0.38, [
    #    "1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24",
    #    "25","26","27","28","29","30","31","32"])
    plt.legend()
    plt.show()

   
 ###########################################################################
# Método para graficar el porcentaje de cada Sector que estuvieron en UCI  #
###########################################################################      
def Columna_sector(df):
    
    TC_SECTOR=pd.crosstab(index=df['UCI'], columns=df['SECTOR']).apply(lambda r: r/r.sum() *100,axis=0)#crostab de UCI y Sector
    TC_SECTOR=np.array(TC_SECTOR)
    dicc_setor=dict(zip(np.array([1,3,4,5,6,8,9,10,11,12]),TC_SECTOR[0])) #Se obtiene el primer renglon  de la tabla y a cada valor 
                                                                          # se le asigna el numero de su sector 
    dicc_setor_ordenado=sorted(dicc_setor.items(), key=operator.itemgetter(1), reverse=True)#se ordena el diccionario de mayor 
                                                                                            #a menor  
    dicc_setor_ordenado_D=dict(dicc_setor_ordenado)
    X = np.arange(len(dicc_setor_ordenado_D))
    
    #Se grafica el diccionario ordenado
    pl.bar(X, dicc_setor_ordenado_D.values(), align='center', width=0.5)
    pl.xticks(X, dicc_setor_ordenado_D.keys())
    ymax = max(dicc_setor_ordenado_D.values()) + 1
    pl.ylim(0, ymax)
    pl.show() 
    
        
    
    
#######################################################################################
#                                                                                    # 
#Metodo que regresa matriz de 2 por 2 , columnas y filas con valores si y no(1 y 2)  #
######################################################################################   
def parte_tabla(df,band):
    
    t=np.array(df)
    if(band==0):
        nueva=np.array([[t[0][0],t[0][1]],
                        [t[1][0],t[1][1]]])
    if(band==1):
       
         nueva=np.array([[t[0][0],t[0][1],t[0][2],t[0][3],t[0][4],t[0][5]],
                        [t[1][0],t[1][1],t[0][2],t[0][3],t[0][4],t[0][5]]])
        
    return nueva

##########################################################################
# Grafica las edades del todos el conjunto de  datos                     #
##########################################################################
def Columna_edad(df):  
    plot = pd.crosstab(index=df['UCI'],columns=df['EDAD']).apply(lambda r: r/r.sum() *100,axis=1).plot(kind='bar')

    
##########################################################################
# Método que regresa el riego de estar en UCI teniendo x enfremedad      #
##########################################################################
def riesgo_relativo(mat):
    
    num=mat[0][0]/(mat[0][0]+mat[0][1])
    den=mat[1][0]/(mat[1][0]+mat[1][1])    
  
    return num/den


##########################################################################
# Regreda la tabla de contingencia de numpy                              #
##########################################################################

def tabla_de_contingencia(df_cont,col1,col2):
    return pd.crosstab(index=df_cont[col1],columns=df_cont[col2],margins=True)

##########################################################################
# Regreda el porcentaje de  "columna" en UCI                             #
##########################################################################
def porcentaje_de_UCI(mat):
    
    cien=100
    total_variable=mat[0][0]+mat[0][1]
    si_si=mat[0][0]
    ope=(si_si*cien)/total_variable
    
    return ope 

  
    
def Columna_entidad_nac(df_cdmx_SIN_99_97):
    print(pd.value_counts(df_cdmx_SIN_99_97['ENTIDAD_NAC']))  
    plot = df_cdmx['ENTIDAD_NAC'].value_counts().plot(kind='bar',
                                            title='ENTIDAD_NAC')
    
##########################################################################
# lee archivo csv                                                        #
##########################################################################
def leer_archivo(nombre_archivo):
    return pd.read_csv(nombre_archivo, sep=',', engine='python') 


##########################################################################
# Crea Tabla de contingencia                                             #
##########################################################################

def mi_tabla_de_contingencia_2por2(df,col1,col2):  
    
    #len(UCI_otra[(UCI_otra[col1]==1) & (UCI_otra[col2]==1)      -> total de Col1==1 y col2==1
    #len(UCI_otra[(UCI_otra[col1]==1) &   (UCI_otra[col2]==2)]) -> total de Col1==1 y col2==2 
    #len(UCI_otra[(UCI_otra[col1]==2) &   (UCI_otra[col2]==1)]) -> total de Col1==2 y col2==1 
    #len(UCI_otra[(UCI_otra[col1]==2) &   (UCI_otra[col2]==2)]) -> total de Col1==2 y col2==2 
    
    
    
    UCI_otra = df[[col1,col2]]  
    tabla_contingencia=np.array([ [len(UCI_otra[(UCI_otra[col1]==1) & (UCI_otra[col2]==1)]),  len(UCI_otra[(UCI_otra[col1]==1) &   (UCI_otra[col2]==2)])],
                                 [  len(UCI_otra[(UCI_otra[col1]==2) & (UCI_otra[col2]==1)]),  len(UCI_otra[(UCI_otra[col1]==2) & (UCI_otra[col2]==2)])]])
        
    return tabla_contingencia

##########################################################################
# Creo tabla de contingencia para edad                                   #
##########################################################################

def mi_tabla_de_contingenciaEDAD(df,col1,col2):
    
  #len(UCI_otra[(UCI_otra[col1]==1) & (UCI_otra[col2]=='1') -> total de UCI==1 y EDAD==1
  #len(UCI_otra[(UCI_otra[col1]==1) & (UCI_otra[col2]=='2') -> total de UCI==1 y EDAD==2
  #len(UCI_otra[(UCI_otra[col1]==1) & (UCI_otra[col2]=='3') -> total de UCI==1 y EDAD==3 
  #len(UCI_otra[(UCI_otra[col1]==1) & (UCI_otra[col2]=='4') -> total de UCI==1 y EDAD==4  
  #len(UCI_otra[(UCI_otra[col1]==1) & (UCI_otra[col2]=='5') -> total de UCI==1 y EDAD==5  
  #len(UCI_otra[(UCI_otra[col1]==1) & (UCI_otra[col2]=='6') -> total de UCI==1 y EDAD==6
    
  #len(UCI_otra[(UCI_otra[col1]==2) & (UCI_otra[col2]=='1') -> total de UCI==2 y EDAD==1
  #len(UCI_otra[(UCI_otra[col1]==2) & (UCI_otra[col2]=='2') -> total de UCI==2 y EDAD==2
  #len(UCI_otra[(UCI_otra[col1]==2) & (UCI_otra[col2]=='3') -> total de UCI==2 y EDAD==3 
  #len(UCI_otra[(UCI_otra[col1]==2) & (UCI_otra[col2]=='4') -> total de UCI==2 y EDAD==4  
  #len(UCI_otra[(UCI_otra[col1]==2) & (UCI_otra[col2]=='5') -> total de UCI==2 y EDAD==5  
  #len(UCI_otra[(UCI_otra[col1]==2) & (UCI_otra[col2]=='6') -> total de UCI==2 y EDAD==6
   

    UCI_otra = df[[col1,col2]]  
    tabla_contingencia=np.array([ [len(UCI_otra[(UCI_otra[col1]==1) & (UCI_otra[col2]=='1')]),  len(UCI_otra[(UCI_otra[col1]==1) & (UCI_otra[col2]=='2')]),
                                  len(UCI_otra[(UCI_otra[col1]==1) & (UCI_otra[col2]=='3')]),len(UCI_otra[(UCI_otra[col1]==1) & (UCI_otra[col2]=='4')]),
                                   len(UCI_otra[(UCI_otra[col1]==1) & (UCI_otra[col2]=='5')]),len(UCI_otra[(UCI_otra[col1]==1) & (UCI_otra[col2]=='6')])
                                  
                                  ],                                 
                                 
                                 [len(UCI_otra[(UCI_otra[col1]==2) & (UCI_otra[col2]=='1')]),  len(UCI_otra[(UCI_otra[col1]==2) & (UCI_otra[col2]=='2')]),
                                   len(UCI_otra[(UCI_otra[col1]==2) & (UCI_otra[col2]=='3')]),  len(UCI_otra[(UCI_otra[col1]==2) & (UCI_otra[col2]=='4')]),
                                   len(UCI_otra[(UCI_otra[col1]==2) & (UCI_otra[col2]=='5')]),  len(UCI_otra[(UCI_otra[col1]==2) & (UCI_otra[col2]=='6')])
                                                            
                                 
                                 ]]) 

    return tabla_contingencia


##########################################################################
# Métodos para calcular dependencia o independencia de dos variables
##########################################################################
def chi_cuadrada(obs):  

    prob=0.95
    stat,p_val,dof,exp_val=chi2_contingency(obs) 
    alpha = 1.0 - prob
    
    if p_val <= alpha:
        return 'Dependiente'
    else:
        return 'Independiente' 
def fisher(obs):
   
    prob=0.95
    alpha = 1.0 - prob
    oddsratio, pvalue = stats.fisher_exact(obs)   
  
    if pvalue <= alpha:
        return 'Dependiente'
    else:
        return 'Independiente'   
    
##########################################################################
# Información del archivo
##########################################################################    
def info_dataFrame(df):     
    print("tamaño ",len(df))
    print("total columnas",len(df.dtypes))
    print("Tipos de columnas")
    print(df.dtypes)
    print(df.head())  
    
##########################################################################
# Método para eliminar columans de DF
##########################################################################      
def eliminar_algunas_columnas(df,algunas_columnas):
    return df.drop(algunas_columnas, axis=1) 



##########################################################################
# Método que drafica  enferemedades o padecimientos de un estado         #
##########################################################################
def Graficar_enfermedades(df,estado):   
    columnas_enf=['INTUBADO','NEUMONIA','DIABETES','EPOC','ASMA','INMUSUPR','HIPERTENSION',
                                   'OTRA_COM','CARDIOVASCULAR','OBESIDAD','RENAL_CRONICA','TABAQUISMO','OTRO_CASO']
    
    porc,li,od,fr=asociar_variables_con_UCI(df,columnas_enf,'UCI')
    
    #Garfico 3 metricas para compararlas
    graficar_metrica(porc,"porcentaje",estado)    
    graficar_metrica(fr,"frecuencia",estado)        
    graficar_metrica(li,"ration",estado)
   # graficar_metrica(od,"ODD ration",estado)

def Odds_ratio(mat):      
    num=mat[0][0]*mat[1][1]
    den=mat[1][0]*mat[0][1]
   
    return num/den



##########################################################################
# Método que regresa diccionario ordenado  de mayor a menor de           #
#enferemedades o padecimientos                                           #
##########################################################################

def asociar_variables_con_UCI(df_3,nombre_columnas,UCI):
    
    li=[]
    FR=[]
    od=[]
    porc=[]
    
    for i in range (len(nombre_columnas)):   
        
        tablita=parte_tabla(tabla_de_contingencia(df_3,UCI,nombre_columnas[i]),0)#tabla de contingencia          
        rr=riesgo_relativo(tablita)#calcula riesgo relativo de una columna con UCI 
        chi=chi_cuadrada(tablita)
        oo=Odds_ratio(tablita)       
        v_por=porcentaje_de_UCI(tablita)   #calcula porcentaje de una columna con UCI     
        
        li.append(round(rr,6))
        FR.append(tablita[0][0])      
        od.append(round(oo,6))      
        porc.append(round(v_por,6)) 
        
    return dict(zip(nombre_columnas,porc)),dict(zip(nombre_columnas,li)),dict(zip(nombre_columnas,od)),dict(zip(nombre_columnas,FR))       

 
##################################################################################
# Método que grafica ['EMBARAZO','HABLA_LENGUA_INDIG','RESULTADO'] de un estado  # 
# para ver cual esta mas relacionada con UCI                                     #
##################################################################################    
def asociar_otros(df,estado):
    
    nombre_columnas_otros=['EMBARAZO','HABLA_LENGUA_INDIG','RESULTADO']
    porc,li,od,fr=asociar_variables_con_UCI(df,nombre_columnas_otros,'UCI')
   
    graficar_metrica(porc,"porcentaje",estado)
    graficar_metrica(fr,"frecuencia",estado)    
    graficar_metrica(li,"riesfo relativo",estado)
   # graficar_metrica(od,"odd ration",estado)
    
    
#############################################################################    
# Método para agrupar las edades en 6 grupos
# 1 reprecenta edades de 0-5
# 2 reprecenta edades de 6-12
# 3 reprecenta edades de 13-18
# 4 reprecenta edades de 19-35
# 5 reprecenta edades de 36-60
# 6 reprecenta edades de 60-100
############################################################################
def agrupar_edades(df):
    
    bins=[0,5,12,18,35,60,100]
    names=["1","2","3","4","5","6"]
    df['EDAD']=pd.cut(df['EDAD'],bins,labels=names)
    return df

def UCI_1_2(df):             
    return df[(df.UCI<3)]
    

########################################################################
# Método para graficar una columna en los 32 estados
########################################################################
def graficar_metrica(diccio_metrica,nombre_metrica,estado):
    
    porc_ordenado=sorted(diccio_metrica.items(), key=operator.itemgetter(1), reverse=True)
    porc_ordenado_D=dict(porc_ordenado)
    lenguajes =  porc_ordenado_D.keys()
    y_pos = np.arange(len(lenguajes))
    cantidad_usos = porc_ordenado_D.values()

    plt.barh(y_pos, cantidad_usos, align='center', alpha=0.5)
    plt.yticks(y_pos, lenguajes)
    plt.xlabel('')
    plt.title(estado+" "+ nombre_metrica)
    ima_porc=estado+nombre_metrica+".png"    
    plt.savefig(ima_porc)    
    plt.show()    
    
    
    
    
    











    
    