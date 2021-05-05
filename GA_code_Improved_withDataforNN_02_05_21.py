import random
import os
import numpy as np
import shutil
import sys
from abaqus import *
from abaqusConstants import *
import matplotlib.pyplot as plt
import pickle


def PrintToScreen(Message):
    print >> sys.__stdout__, '%s' % Message


# path = "C:/Users/brend/Desktop/G_A_a_NON_UNIFORM_MUTATION_second"
path = "C:/Users/Luisantos090/Desktop/Abaqus_Projects/Brenda_PlateGirders/Final_code_improved_02-05-2021"
os.chdir(path)

# Range according to UB section tables
height_range = [450.0,900.0,6]
# 4.0 min
webthick_range = [10.0,25.0,4]
top_width_range = [250.0,320.0,4]
#7.6 min
topthick_range = [20.0,50.0,5]
bot_width_range = [75.0,200.0,5]
#7.6 min
botthick_range = [10.0,25.0,4]

Population_size = 100
# Make sure Elitism_size is an even number
Elitism_size = 2
Generations = 50
Tournament_size = 2

variables = [height_range,webthick_range,
             top_width_range,topthick_range,
             bot_width_range,botthick_range]
             
             
# I had to create an hack to keep running each generation everytime abaqus colapses
# We need to go to the folder and delete the beam that caused issues 


PrintToScreen('New Run[write "N"]? Or Continue Previous Run[write "C"]?')
GA_status = input('New Run[write "N"]? Or Continue Previous Run[write "C"]?')
if GA_status == "N":
    GA_status = "New"
elif GA_status == "C":
    GA_status = "Continue"


os.chdir(path)  

Round = 1

if GA_status == "Continue":
    Round = 1

    while os.path.isdir("%s/Round%s" % (path, Round+1)):
        Round = Round + 1
    os.chdir("%s/Round%s" % (path, Round))
    for dirs in os.listdir("%s/Round%s" % (path, Round)):
        if dirs.startswith("[") and os.path.isfile("%s/Round%s/%s/FitnessData.txt" % (path, Round, dirs)) == False:
            shutil.rmtree("%s/Round%s/%s" % (path, Round, dirs))
            break
        
    
def CreateFolder(name):
    if not os.path.exists("%s" % name):
        os.makedirs("%s" % name)
    os.chdir("%s" % name)


def CreateInitialPop(variables, Population_size):
    count = 0 
    InitPop = []
    Chromossome_Lenght = 0
    for i in variables:
        Chromossome_Lenght = Chromossome_Lenght + i[2]
    while count < Population_size:
        Chromossome = ''
        for i in range(Chromossome_Lenght):
            Chromossome = Chromossome + str(random.choice([0,1]))        
        if Chromossome not in InitPop:
            InitPop.append([Chromossome])
            count = count + 1
        else:
            print("------REPEATED-------")
    
    return InitPop


def ChromossomeToData(Chromossome, variables):
    var_data = []
    start = 0
    for i in variables:
        var_binary = Chromossome[0][start:start+i[2]]
        var_decimal = i[0] + int(str(var_binary), 2)*i[3]
        var_data.append(round(var_decimal,5))
        start = start + i[2]
    return var_data


def NewPopulation(OldPop, select, Tournament_size, Current_Round, Generations):
    Newpop = []
    counter = select
    Populationsize = len(OldPop) 
    OldPop_sorted = sorted(OldPop,key=lambda x: x[-1])
    # This does elitism selection
    for i in range(select):
        Newpop.append([OldPop_sorted[i][0]])
    # This does tournament selection before crossover
    while counter < Populationsize:
        father = Tournament(OldPop, Tournament_size)
        mother = Tournament(OldPop, Tournament_size)
        # while mother == father:
            # mother = Tournament(OldPop, fitness, 3)
        child1 = ''
        child2 = ''
        # This does crossover of genes
        if np.random.uniform(0, 1) <= 0.7:
            for i in range(len(OldPop[0][0])):    
                child1 = child1 + str(np.random.choice([father[i],mother[i]]))
                child2 = child2 + str(np.random.choice([father[i],mother[i]]))
        else:
            child1 = father
            child2 = mother
        counter = counter + 2
        # This does mutation of genes
        child1 = MutateChromossome(child1, 0.1+0.4*(Current_Round/float(Generations)))
        child2 = MutateChromossome(child2, 0.1+0.4*(Current_Round/float(Generations)))
        Newpop.append([child1])
        Newpop.append([child2])
    return Newpop


def MutateChromossome(Chromossome, Probability):
    if np.random.uniform(0, 1) <= Probability:
        Random_Gene = np.random.randint(0,len(Chromossome))
        if Chromossome[Random_Gene] == '0':
            Chromossome = Chromossome[0:Random_Gene] + '1' + Chromossome[Random_Gene+1:]
        else:
            Chromossome = Chromossome[0:Random_Gene] + '0' + Chromossome[Random_Gene+1:]
        
    return Chromossome


def Tournament(OldPop, PlayersNumber):
    ww = np.random.randint(0,len(OldPop))
    winner = OldPop[ww][0]
    counter = 0
    while counter < PlayersNumber - 1:
        cc = np.random.randint(0,len(OldPop))
        challenger = OldPop[cc][0]
        #Remember to verify that the Fitness is the last value
        if OldPop[ww][-1] < OldPop[cc][-1]:
            continue
        else:
            ww = cc
            winner = challenger
        counter = counter + 1
    return winner


def SaveInExcel(Foldername, filename, X_data, Y_data):
    
    opFile = Foldername+'/'+filename+'-ForceDispCurve.csv'
    
    try:
        opFileU = open(opFile,'w')
        opFileU.write("%10s,%10s\n"%('Disp','Force') )
    except IOError:
        print 'cannot open', opFile
        exit(0)

    for i in range(len(X_data)):
        opFileU.write("%10f,%10f\n"%(X_data[i][1], Y_data[i]))
        
    opFileU.close()


def MakePlot(filename, bbbbbbb, Forces):
    
    Displacements = [num[1] for num in bbbbbbb]
    
    
    fig, ax = plt.subplots()
    ax.plot(Displacements, Forces)

    ax.set(xlabel='Displacements (mm)', ylabel='Linear Load (N/mm)',
           title='Force Displacement Curve')
    ax.grid()

    fig.savefig("%s.png"%filename)
    plt.close(fig)

    
def GeneticAlgorithm(variables, Generations, Population_size, Elitism_size, Tournament_size, Round, GA_status):
    if GA_status == "New":
        DeleteRounds("%s" % path)
        os.chdir(path)
        Population_binary = CreateInitialPop(variables, Population_size)
        Process = []
        SaveAnalysisbinary = dict()
        
    if GA_status == "Continue":
        os.chdir(path)
        PrintToScreen(Round)
        if Round == 1:
            Process = []
        else:
            Process = pickle.load(open("Process_list", 'rb'))
        os.chdir("%s/Round%s" % (path,Round))  
        Population_binary = pickle.load(open("Population_binary_list", 'rb'))
        SaveAnalysisbinary = pickle.load(open("SaveAnalysisbinary", 'rb'))
        os.chdir(path)
    
    for i in variables:
        step = (i[1]-i[0])/(2**i[2]-1)
        i.append(step)
        
    weight_max = variables[0][1]*variables[1][1] + variables[2][1]*variables[3][1] + variables[4][1]*variables[5][1]
    weight_min = variables[0][0]*variables[1][0] + variables[2][0]*variables[3][0] + variables[4][0]*variables[5][0]
    weight_average = (weight_max + weight_min)/2
    
    for i in range(Round, Generations+1):
        CreateFolder("Round%s" % (i))
        PrintToScreen('-------ROUND %s-------' % (i))
        os.chdir("%s/Round%s" % (path,i))
        pickle.dump(Population_binary, open("Population_binary_list", 'wb'))
        for j in range(Population_size):
            Population_binary[j].append(ChromossomeToData(Population_binary[j], variables))
        for Chromossome in Population_binary:
            # print(Chromossome)
            if str(Chromossome[0]) in SaveAnalysisbinary:
                PrintToScreen("Saving analysis")
                Weight = SaveAnalysisbinary[str(Chromossome[0])][-2]
                Fitness = SaveAnalysisbinary[str(Chromossome[0])][-1]
                Chromossome.append(Weight)
                Chromossome.append(Fitness)
                os.chdir("%s/Round%s" % (path,i))
            else:
                PrintToScreen("New binary")
                CreateFolder("%s" % (Chromossome[1]))
                os.chdir("%s/Round%s/%s" % (path,i,Chromossome[1]))
                Weight, Fitness = FitnessFunction(Chromossome[1], weight_average)
                SaveAnalysisbinary[str(Chromossome[0])] = [Chromossome[1], (i), Weight, Fitness]
                Chromossome.append(Weight)
                Chromossome.append(Fitness)
                os.chdir("%s/Round%s" % (path,i))
                #to save object
                pickle.dump(SaveAnalysisbinary, open("SaveAnalysisbinary", 'wb'))
        os.chdir(path)
        MinFitness = 1.0e8 
        SumFitness = 0
        for Chromossome in Population_binary:
            SumFitness = SumFitness + Chromossome[-1]
            MinFitness = min(Chromossome[-1], MinFitness)
        Process.append([i,MinFitness,SumFitness/Population_size])
        PrintData("%s/Round%s" % (path,i), 'Process', "Round\tMinFitness\tAverageFitness", Process)
        PrintData("%s/Round%s" % (path,i), 'Population_binary', "GeneticCode\tVariables\tWeight\tFitness", Population_binary)
        PrintFinalData("%s/Round%s" % (path,i), 'AllIndividuals', "GeneticCode\tVariables\tRound\tFitness", SaveAnalysisbinary)
        pickle.dump(Process, open("Process_list", 'wb'))
        if i == Generations-1:
            continue
        else:
            Population_binary = NewPopulation(Population_binary, Elitism_size, Tournament_size, i, Generations)
    
    PrintData("%s/Round%s" % (path,i), 'Process', "Round\tMinFitness\tAverageFitness", Process)
    PrintFinalData("%s/Round%s" % (path,i), 'AllIndividuals', "GeneticCode\tVariables\tRound\tFitness", SaveAnalysisbinary)
    PrintToScreen('Brenda almost did it')


def FitnessFunction(variables, weight_average):
    import section
    import regionToolset
    import displayGroupMdbToolset as dgm
    import part
    import material
    import assembly
    import step
    import interaction
    import load
    import mesh
    import optimization
    import job
    import sketch
    import visualization
    import xyPlot
    import displayGroupOdbToolset as dgo
    import connectorBehavior
    
    # VARIABLES OR ARGUMENTS

    # CROSS-SECTION SIZE
    web_depth = variables[0]
    topflange_b1 = variables[2]
    bottomflange_b2 = variables[4]

    # CROSS-SECTION THICKNESS
    webthickness_tw = variables[1]
    topflangethickness_tf1 = variables[3]
    bottomflangethickness_tf2 = variables[5]
    
    # BeamFolder = "/%.1f_%.1f_%.1f_%.1f_%.1f_%.1f" % (web_depth, webthickness_tw, topflange_b1, topflangethickness_tf1, bottomflange_b2, bottomflangethickness_tf2)
    # os.mkdir(path+"/"+str(foldername)+"/"+BeamFolder)
    # os.chdir(path+"/"+str(foldername)+"/"+BeamFolder)
    BeamFolder = os.getcwd()
    
    #SPAN
    beam_span = 7500.0
    
    #scale factor
    scale_factor = beam_span/500.0
    
    # LOAD
    Design_load = 150.0 # Factored ULS Combination (1.35G + 1.5Q)
    Accidental_load = Design_load*1.25 # Acidental Load
    Serviceability_load = Design_load*0.7 # SLS Combination (1.0G + 1.0Q)
    
    UDL = 1.5*(Design_load)       #This is in kN/m same as N/mm

    # STEP
    increments_steps = 100
    initial_steps = 0.01
    max_steps = 0.01

    # MESH
    mesh_size = 50.0

    # BEAM LENGHT
    beam_lenght = beam_span+bottomflange_b2
  
    # Create new model
    Mdb()
    
    
    # PART: SHELL BY EXTRUCTION
    
    s = mdb.models['Model-1'].ConstrainedSketch(name='__profile__', 
        sheetSize=1000.0)
    g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
    s.setPrimaryObject(option=STANDALONE)
    
    
    # HEIGHT OF THE WEB, web_depth.
    
    s.Line(point1=(0.0, -web_depth/2), point2=(0.0, web_depth/2))
    s.VerticalConstraint(entity=g[2], addUndoState=False)
    
    
    # LENGHT OF THE TOP FLANGE, b1.
    
    s.Line(point1=(-topflange_b1/2, web_depth/2), point2=(topflange_b1/2, web_depth/2))
    s.HorizontalConstraint(entity=g[3], addUndoState=False)
    
    
    # LENGHT OF THE BOTTOM FLANGE, b2.
    
    s.Line(point1=(-bottomflange_b2/2, -web_depth/2), point2=(bottomflange_b2/2, -web_depth/2))
    s.HorizontalConstraint(entity=g[4], addUndoState=False)
    
    
    # LENGHT OF THE BEAM
    
    p = mdb.models['Model-1'].Part(name='I_Section', dimensionality=THREE_D, 
        type=DEFORMABLE_BODY)
    p = mdb.models['Model-1'].parts['I_Section']
    p.BaseShellExtrude(sketch=s, depth=beam_lenght)
    s.unsetPrimaryObject()
    p = mdb.models['Model-1'].parts['I_Section']
    
    del mdb.models['Model-1'].sketches['__profile__']
    
    
    # MATERIAL PROPERTIES FOR THE I BEAM
    
    mdb.models['Model-1'].Material(name='Steel')
    mdb.models['Model-1'].materials['Steel'].Elastic(table=((210000.0, 0.3), ))
    # mdb.models['Model-1'].materials['Steel'].Plastic(table=((355.0, 0.0), ))    
    # mdb.models['Model-1'].materials['Steel'].Plastic(table=((355.0, 0.0), (510.0, 0.2)))
    
    
    # TOP FLANGE WIDTH, tf1.
    
    mdb.models['Model-1'].HomogeneousShellSection(name='tf1', preIntegrate=OFF, 
        material='Steel', thicknessType=UNIFORM, thickness=topflangethickness_tf1, 
        thicknessField='', nodalThicknessField='', 
        idealization=NO_IDEALIZATION, poissonDefinition=DEFAULT, 
        thicknessModulus=None, temperature=GRADIENT, useDensity=OFF, 
        integrationRule=SIMPSON, numIntPts=5)
    
    
    # WEB WIDTH, tw.
    
    mdb.models['Model-1'].HomogeneousShellSection(name='tw', preIntegrate=OFF, 
        material='Steel', thicknessType=UNIFORM, thickness=webthickness_tw, 
        thicknessField='', nodalThicknessField='', 
        idealization=NO_IDEALIZATION, poissonDefinition=DEFAULT, 
        thicknessModulus=None, temperature=GRADIENT, useDensity=OFF, 
        integrationRule=SIMPSON, numIntPts=5)
    
    
    # SPAN - BOTTOM FLANGE WIDTH, tf2_span.
    
    mdb.models['Model-1'].HomogeneousShellSection(name='tf2_span', preIntegrate=OFF, 
        material='Steel', thicknessType=UNIFORM, thickness=bottomflangethickness_tf2, 
        thicknessField='', nodalThicknessField='', 
        idealization=NO_IDEALIZATION, poissonDefinition=DEFAULT, 
        thicknessModulus=None, temperature=GRADIENT, useDensity=OFF, 
        integrationRule=SIMPSON, numIntPts=5)
    
    # MATERIAL FOR SUPPORTS
    
    mdb.models['Model-1'].Material(name='Stiff_support')
    mdb.models['Model-1'].materials['Stiff_support'].Elastic(table=((2.1e+8, 0.3), ))
    # mdb.models['Model-1'].materials['Stiff_support'].Plastic(table=((355.0, 0.0), ))
    # mdb.models['Model-1'].materials['Stiff_support'].Plastic(table=((355.0, 0.0), (510.0, 0.2)))
    
    
    # SUPPORT - BOTTOM FLANGE WIDTH, tf2.
    
    mdb.models['Model-1'].HomogeneousShellSection(name='tf2_support', 
        preIntegrate=OFF, material='Stiff_support', thicknessType=UNIFORM, 
        thickness=bottomflangethickness_tf2, thicknessField='', nodalThicknessField='', 
        idealization=NO_IDEALIZATION, poissonDefinition=DEFAULT, 
        thicknessModulus=None, temperature=GRADIENT, useDensity=OFF, 
        integrationRule=SIMPSON, numIntPts=5)
    
    
    # CONVERGING
    

    # DATUM OFFSET FROM PINNED SUPPORT for B.C.
    p = mdb.models['Model-1'].parts['I_Section']
    p.DatumPlaneByPrincipalPlane(principalPlane=XYPLANE, offset=bottomflange_b2/2)
    
    # DATUM OFFSET FROM ROLLER SUPPORT for B.C.
    p = mdb.models['Model-1'].parts['I_Section']
    p.DatumPlaneByPrincipalPlane(principalPlane=XYPLANE, offset=beam_lenght-bottomflange_b2/2)
    
    # DATUM OFFSET FROM PINNED SUPPORT for plate
    p = mdb.models['Model-1'].parts['I_Section']
    p.DatumPlaneByPrincipalPlane(principalPlane=XYPLANE, offset=bottomflange_b2)
    
    # DATUM OFFSET FROM ROLLER SUPPORT for plate
    p = mdb.models['Model-1'].parts['I_Section']
    p.DatumPlaneByPrincipalPlane(principalPlane=XYPLANE, offset=beam_lenght-bottomflange_b2)
    
    
    # PLANE-CUT FROM PINNED SUPPORT for B.C.
    p = mdb.models['Model-1'].parts['I_Section']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#1f ]', ), )
    d1 = p.datums
    p.PartitionFaceByDatumPlane(datumPlane=d1[2], faces=pickedFaces)
    
    # PLANE-CUT FROM ROLLER SUPPORT for B.C.
    p = mdb.models['Model-1'].parts['I_Section']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#2aa ]', ), )
    d2 = p.datums
    p.PartitionFaceByDatumPlane(datumPlane=d2[3], faces=pickedFaces)
    
    # PLANE-CUT FROM PINNED SUPPORT for plate
    p = mdb.models['Model-1'].parts['I_Section']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#18 ]', ), )
    d1 = p.datums
    p.PartitionFaceByDatumPlane(datumPlane=d1[4], faces=pickedFaces)
    
    # PLANE-CUT FROM ROLLER SUPPORT for plate
    p = mdb.models['Model-1'].parts['I_Section']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#21 ]', ), )
    d2 = p.datums
    p.PartitionFaceByDatumPlane(datumPlane=d2[5], faces=pickedFaces)
    
    
    # ASSIGNATION OF SECTION tf1
    
    p = mdb.models['Model-1'].parts['I_Section']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#18c30 ]', ), )
    region = regionToolset.Region(faces=faces)
    p = mdb.models['Model-1'].parts['I_Section']
    p.SectionAssignment(region=region, sectionName='tf1', offset=0.0, 
        offsetType=TOP_SURFACE, offsetField='', 
        thicknessAssignment=FROM_SECTION)
    
    
    # ASSIGNATION OF SECTION tw
    
    p = mdb.models['Model-1'].parts['I_Section']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#6040 ]', ), )
    region = regionToolset.Region(faces=faces)
    p = mdb.models['Model-1'].parts['I_Section']
    p.SectionAssignment(region=region, sectionName='tw', offset=0.0, 
        offsetType=MIDDLE_SURFACE, offsetField='', 
        thicknessAssignment=FROM_SECTION)
    
    
    # ASSIGNATION OF SECTION tf2_span
    
    p = mdb.models['Model-1'].parts['I_Section']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#3 ]', ), )
    region = regionToolset.Region(faces=faces)
    p = mdb.models['Model-1'].parts['I_Section']
    p.SectionAssignment(region=region, sectionName='tf2_span', offset=0.0, 
        offsetType=BOTTOM_SURFACE, offsetField='', 
        thicknessAssignment=FROM_SECTION)
    
    
    # ASSIGNATION OF SECTION tf2_support
    
    p = mdb.models['Model-1'].parts['I_Section']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#6138c ]', ), )
    region = regionToolset.Region(faces=faces)
    p = mdb.models['Model-1'].parts['I_Section']
    p.SectionAssignment(region=region, sectionName='tf2_support', offset=0.0, 
        offsetType=BOTTOM_SURFACE, offsetField='', 
        thicknessAssignment=FROM_SECTION)
    
    
    # SETS for B.C. for PINNED SUPPORT
    
    
    # SET --- Internal lines
    
    p = mdb.models['Model-1'].parts['I_Section']
    e = p.edges
    edges = e.getSequenceFromMask(mask=('[#4000800 ]', ), )
    p.Set(edges=edges, name='B_C_1_Int_line')
    
    
    # SET for restrain WEB --- Internal lines
    
    p = mdb.models['Model-1'].parts['I_Section']
    e = p.edges
    edges = e.getSequenceFromMask(mask=('[#400000 ]', ), )
    p.Set(edges=edges, name='Rest_Y_1_Int_line')
    
    
    # SET for restrain FLANGE --- Point
    
    p = mdb.models['Model-1'].parts['I_Section']
    v = p.vertices
    verts = v.getSequenceFromMask(mask=('[#2000 ]', ), )
    p.Set(vertices=verts, name='Rest_Z_1_Point')
    
    
    
    # SETS for B.C. for ROLLER SUPPORT
    
    
    # SET --- Internal lines
    
    p = mdb.models['Model-1'].parts['I_Section']
    e = p.edges
    edges = e.getSequenceFromMask(mask=('[#1000100 ]', ), )
    p.Set(edges=edges, name='B_C_2_Int_line')
    
    
    # SET for restrain WEB --- Internal lines
    
    p = mdb.models['Model-1'].parts['I_Section']
    e = p.edges
    edges = e.getSequenceFromMask(mask=('[#200000 ]', ), )
    p.Set(edges=edges, name='Rest_Y_2_Int_line')
    
    
    # SET for restrain FLANGE --- Point
    
    p = mdb.models['Model-1'].parts['I_Section']
    v = p.vertices
    verts = v.getSequenceFromMask(mask=('[#4000 ]', ), )
    p.Set(vertices=verts, name='Rest_Z_2_Point')
    
    
    
    
    
    
    
    # INDEPENDENT ASSEMBLY
    
    a = mdb.models['Model-1'].rootAssembly
    a.DatumCsysByDefault(CARTESIAN)
    p = mdb.models['Model-1'].parts['I_Section']
    a.Instance(name='I_Section-1', part=p, dependent=OFF)
     
    
    # ROTATE AXIS
    
    a = mdb.models['Model-1'].rootAssembly
    a.rotate(instanceList=('I_Section-1', ), axisPoint=(0.0, -(web_depth/2), 0.0), 
        axisDirection=(0.0, web_depth, 0.0), angle=90.0)

    
    # EIGEN MODE - DATUM PLANES -

    
    ##############            Displacement planes             ################
    
    a = mdb.models['Model-1'].rootAssembly
    a.DatumPlaneByPrincipalPlane(principalPlane=YZPLANE, offset=beam_lenght/2)
    
    
  
    ##############            Shear buckling planes             ################
    
    a = mdb.models['Model-1'].rootAssembly
    a.DatumPlaneByPrincipalPlane(principalPlane=XZPLANE, offset=0.0)
    
    a = mdb.models['Model-1'].rootAssembly
    a.DatumPlaneByPrincipalPlane(principalPlane=YZPLANE, offset=beam_lenght*0.10)
    
    a = mdb.models['Model-1'].rootAssembly
    a.DatumPlaneByPrincipalPlane(principalPlane=YZPLANE, offset=beam_lenght*0.18)
    
    a = mdb.models['Model-1'].rootAssembly
    a.DatumPlaneByPrincipalPlane(principalPlane=YZPLANE, offset=beam_lenght*0.82)
    
    a = mdb.models['Model-1'].rootAssembly
    a.DatumPlaneByPrincipalPlane(principalPlane=YZPLANE, offset=beam_lenght*0.90)
    
    
    
    ##############        Lateral torsional and Compression buckling planes        ################
    
    a = mdb.models['Model-1'].rootAssembly
    a.DatumPlaneByPrincipalPlane(principalPlane=YZPLANE, offset=beam_lenght*0.42)
    
    a = mdb.models['Model-1'].rootAssembly
    a.DatumPlaneByPrincipalPlane(principalPlane=YZPLANE, offset=beam_lenght*0.482)
    
    a = mdb.models['Model-1'].rootAssembly
    a.DatumPlaneByPrincipalPlane(principalPlane=YZPLANE, offset=beam_lenght*0.542)
    
    a = mdb.models['Model-1'].rootAssembly
    a.DatumPlaneByPrincipalPlane(principalPlane=YZPLANE, offset=beam_lenght*0.604)
    
    
    
    
    
    ##############          partitions  with the extra line close to B.C.      ################
    
    a = mdb.models['Model-1'].rootAssembly
    f1 = a.instances['I_Section-1'].faces
    pickedFaces = f1.getSequenceFromMask(mask=('[#7ffff ]', ), )
    d1 = a.datums
    a.PartitionFaceByDatumPlane(datumPlane=d1[4], faces=pickedFaces)
    a = mdb.models['Model-1'].rootAssembly
    f1 = a.instances['I_Section-1'].faces
    pickedFaces = f1.getSequenceFromMask(mask=('[#ffffff ]', ), )
    d11 = a.datums
    a.PartitionFaceByDatumPlane(datumPlane=d11[6], faces=pickedFaces)
    a = mdb.models['Model-1'].rootAssembly
    f1 = a.instances['I_Section-1'].faces
    pickedFaces = f1.getSequenceFromMask(mask=('[#1fffffff ]', ), )
    d1 = a.datums
    a.PartitionFaceByDatumPlane(datumPlane=d1[7], faces=pickedFaces)
    a = mdb.models['Model-1'].rootAssembly
    f1 = a.instances['I_Section-1'].faces
    pickedFaces = f1.getSequenceFromMask(mask=('[#ffffffff #3 ]', ), )
    d11 = a.datums
    a.PartitionFaceByDatumPlane(datumPlane=d11[8], faces=pickedFaces)
    a = mdb.models['Model-1'].rootAssembly
    f1 = a.instances['I_Section-1'].faces
    pickedFaces = f1.getSequenceFromMask(mask=('[#ffffffff #7f ]', ), )
    d1 = a.datums
    a.PartitionFaceByDatumPlane(datumPlane=d1[9], faces=pickedFaces)
    a = mdb.models['Model-1'].rootAssembly
    f1 = a.instances['I_Section-1'].faces
    pickedFaces = f1.getSequenceFromMask(mask=('[#ffffffff #fff ]', ), )
    d11 = a.datums
    a.PartitionFaceByDatumPlane(datumPlane=d11[10], faces=pickedFaces)
    a = mdb.models['Model-1'].rootAssembly
    f1 = a.instances['I_Section-1'].faces
    pickedFaces = f1.getSequenceFromMask(mask=('[#ffffffff #1ffff ]', ), )
    d1 = a.datums
    a.PartitionFaceByDatumPlane(datumPlane=d1[11], faces=pickedFaces)
    a = mdb.models['Model-1'].rootAssembly
    f1 = a.instances['I_Section-1'].faces
    pickedFaces = f1.getSequenceFromMask(mask=('[#ffffffff #3fffff ]', ), )
    d11 = a.datums
    a.PartitionFaceByDatumPlane(datumPlane=d11[12], faces=pickedFaces)
    a = mdb.models['Model-1'].rootAssembly
    f1 = a.instances['I_Section-1'].faces
    pickedFaces = f1.getSequenceFromMask(mask=('[#ffffffff #7ffffff ]', ), )
    d1 = a.datums
    a.PartitionFaceByDatumPlane(datumPlane=d1[13], faces=pickedFaces)
    a = mdb.models['Model-1'].rootAssembly
    f1 = a.instances['I_Section-1'].faces
    pickedFaces = f1.getSequenceFromMask(mask=('[#ffffffff:2 ]', ), )
    d1 = a.datums
    a.PartitionFaceByDatumPlane(datumPlane=d1[5], faces=pickedFaces)
    
    
    
    # CREATE STEP - STATIC ANALYSIS
    
    # mdb.models['Model-1'].StaticStep(name='Static_Analysis_UDL', previous='Initial', 
        # maxNumInc=increments_steps, initialInc=initial_steps, maxInc=max_steps, nlgeom=ON)
    
    
    
    # CREATE STEP - BUCKLING ANALYSIS

    # mdb.models['Model-1'].StaticRiksStep(name='Step-1', previous='Initial',
    # maxNumInc=increments_steps, initialArcInc=0.01, nlgeom=ON)
    
    mdb.models['Model-1'].BuckleStep(name='Step-1', previous='Initial', 
        numEigen=3, vectors=8, maxIterations=300)
    
 
    
    # BOUNDARY CONDITION for PINNED SUPPORT
    
    
    # PINNED SUPPORT STIFF BASE
    
    # B_C_1 middle line
    
    a = mdb.models['Model-1'].rootAssembly
    region = a.instances['I_Section-1'].sets['B_C_1_Int_line']
    mdb.models['Model-1'].DisplacementBC(name='B_C_1_Int_line', 
        createStepName='Initial', region=region, u1=SET, u2=SET, u3=SET, 
        ur1=SET, ur2=UNSET, ur3=UNSET, amplitude=UNSET, 
        distributionType=UNIFORM, fieldName='', localCsys=None)
    
    
    
    # PINNED SUPPORT RESTRAIN WEB
    
    # Rest_Y_1 internal lines
    
    a = mdb.models['Model-1'].rootAssembly
    region = a.instances['I_Section-1'].sets['Rest_Y_1_Int_line']
    mdb.models['Model-1'].DisplacementBC(name='Rest_Y_1_Int_line', 
        createStepName='Initial', region=region, u1=UNSET, u2=UNSET, u3=SET, 
        ur1=UNSET, ur2=UNSET, ur3=UNSET, amplitude=UNSET, 
        distributionType=UNIFORM, fieldName='', localCsys=None)
    
    
    
    # PINNED SUPPORT RESTRAIN FLANGE
    
    # Rest_Z_1 point
    
    # a = mdb.models['Model-1'].rootAssembly
    # region = a.instances['I_Section-1'].sets['Rest_Z_1_Point']
    # mdb.models['Model-1'].DisplacementBC(name='Rest_Z_1_Point', 
        # createStepName='Initial', region=region, u1=UNSET, u2=UNSET, u3=SET, 
        # ur1=UNSET, ur2=UNSET, ur3=UNSET, amplitude=UNSET, 
        # distributionType=UNIFORM, fieldName='', localCsys=None)
    
    
    
    
    # BOUNDARY CONDITION for ROLLER SUPPORT
    
    
    # ROLLER SUPPORT STIFF BASE
    
    # B_C_2 middle line
    
    a = mdb.models['Model-1'].rootAssembly
    region = a.instances['I_Section-1'].sets['B_C_2_Int_line']
    mdb.models['Model-1'].DisplacementBC(name='B_C_2_Int_line', 
        createStepName='Initial', region=region, u1=SET, u2=SET, u3=SET, 
        ur1=SET, ur2=UNSET, ur3=UNSET, amplitude=UNSET, 
        distributionType=UNIFORM, fieldName='', localCsys=None)
    
    
    
    # ROLLER SUPPORT RESTRAIN WEB
    
    # Rest_Y_2 internal lines
    
    a = mdb.models['Model-1'].rootAssembly
    region = a.instances['I_Section-1'].sets['Rest_Y_2_Int_line']
    mdb.models['Model-1'].DisplacementBC(name='Rest_Y_2_Int_line', 
        createStepName='Initial', region=region, u1=UNSET, u2=UNSET, u3=SET, 
        ur1=UNSET, ur2=UNSET, ur3=UNSET, amplitude=UNSET, 
        distributionType=UNIFORM, fieldName='', localCsys=None)
    
    
    
    # ROLLER SUPPORT RESTRAIN FLANGE
    
    # Rest_Z_2 point
    
    # a = mdb.models['Model-1'].rootAssembly
    # region = a.instances['I_Section-1'].sets['Rest_Z_2_Point']
    # mdb.models['Model-1'].DisplacementBC(name='Rest_Z_2_Point', 
        # createStepName='Initial', region=region, u1=UNSET, u2=UNSET, u3=SET, 
        # ur1=UNSET, ur2=UNSET, ur3=UNSET, amplitude=UNSET, 
        # distributionType=UNIFORM, fieldName='', localCsys=None)
    
    
    
    # for a FIX SUPPORT just set all the degrees of freedom
    # also to change the degree of freedom
    
    
    
    
    
    
    #######    Check   #########      createStepName='Step-1' and write there the analysis to run      #########
    
    
    # LOAD FOR STATIC ANALYSIS = PRESSURE ON TOP OF TOP FLANGE
    
    # a = mdb.models['Model-1'].rootAssembly
    # s1 = a.instances['I_Section-1'].faces
    # side2Faces1 = s1.getSequenceFromMask(mask=('[#18c30 ]', ), )
    # region = regionToolset.Region(side2Faces=side2Faces1)
    # mdb.models['Model-1'].Pressure(name='UDL_load', 
        # createStepName='Static_Analysis_UDL', region=region, 
        # distributionType=UNIFORM, field='', magnitude=UDL)
    
    ################## NEW PRESSURE LOAD #############################################
    
    # a = mdb.models['Model-1'].rootAssembly
    # s1 = a.instances['I_Section-1'].faces
    # side2Faces1 = s1.getSequenceFromMask(mask=('[#18f03000 #614a6303 #318 ]', ), )
    # region = a.Surface(side2Faces=side2Faces1, name='Surf-1')
    # mdb.models['Model-1'].Pressure(name='UDL_load', createStepName='Step-1', 
        # region=region, distributionType=UNIFORM, field='', magnitude=1.0/topflange_b1)
    
    
    
    #################################                   BODY FORCE                 ###################################
    
    a = mdb.models['Model-1'].rootAssembly
    f1 = a.instances['I_Section-1'].faces
    faces1 = f1.getSequenceFromMask(mask=('[#21000556 #80040000 ]', ), )
    region = regionToolset.Region(faces=faces1)
    mdb.models['Model-1'].BodyForce(name='UDL_load', createStepName='Step-1', 
        region=region, comp2=(-1.0/((web_depth/2)*webthickness_tw)))
    
    
    
    
    

    
    # SEED INDEPENDANT ASSEMBLY
    
    a = mdb.models['Model-1'].rootAssembly
    partInstances =(a.instances['I_Section-1'], )
    a.seedPartInstance(regions=partInstances, size=mesh_size, deviationFactor=0.1, 
        minSizeFactor=0.1)
    

    
    # from linear to quadratic assign element type for mesh, increase the nodes
    
    # elemType1 = mesh.ElemType(elemCode=S8R, elemLibrary=STANDARD)
    # elemType2 = mesh.ElemType(elemCode=STRI65, elemLibrary=STANDARD)
    # a = mdb.models['Model-1'].rootAssembly
    # f1 = a.instances['I_Section-1'].faces
    # faces1 = f1.getSequenceFromMask(mask=('[#7ffff ]', ), )
    # pickedRegions =(faces1, )
    # a.setElementType(regions=pickedRegions, elemTypes=(elemType1, elemType2))
    
    
    # MESH INDEPENDANT ASSEMBLY
    
    a = mdb.models['Model-1'].rootAssembly
    partInstances =(a.instances['I_Section-1'], )
    a.generateMesh(regions=partInstances)
    

    
    # copy model 1 to model-2
    
    mdb.Model(name='Model-2', objectToCopy=mdb.models['Model-1'])
    p = mdb.models['Model-2'].parts['I_Section']
    
    
    
    
    # CREATE THE JOB
    
    mdb.Job(name='Job-1', model='Model-1', description='', type=ANALYSIS, 
        atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90, 
        memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True, 
        explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, echoPrint=OFF, 
        modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine='', 
        scratch='', resultsFormat=ODB, multiprocessingMode=DEFAULT, numCpus=1, 
        numGPUs=0)
    
    
    # mdb.Job(name='Job-1', model='Model-1', description='', type=ANALYSIS, 
        # atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90, 
        # memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True, 
        # explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, echoPrint=OFF, 
        # modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine='', 
        # scratch='', resultsFormat=ODB, parallelizationMethodExplicit=DOMAIN, 
        # numDomains=6, activateLoadBalancing=False, multiprocessingMode=DEFAULT, 
        # numCpus=6, numGPUs=1)
    

    
    # open the edit keyword
    
    import job
    mdb.models['Model-1'].keywordBlock.synchVersions(storeNodesAndElements=False)
    
    
    
    # add the field
    
    mdb.models['Model-1'].keywordBlock.replace(73, """
    *Output, field, variable=PRESELECT
    *NODE FILE
    U""")
    
   
    
    # SUBMIT THE JOB

    mdb.jobs['Job-1'].submit(consistencyChecking=OFF)
    
    
    # TO WAIT FOR JOB COMPLETION
    
    mdb.jobs['Job-1'].waitForCompletion()
    print("Job-1 finished running")
    
    
    
    ######################### MODEL 2 #################################################
 
    
    # create the new material with plasticity by copy and delete the old
    
    p1 = mdb.models['Model-2'].parts['I_Section']
   
    mdb.models['Model-2'].Material(name='Steel-nonlinear', 
        objectToCopy=mdb.models['Model-2'].materials['Steel'])
    # mdb.models['Model-2'].materials['Steel-nonlinear'].Plastic(table=((355.0, 0.0), 
        # (510.0, 0.2)))
    mdb.models['Model-2'].materials['Steel-nonlinear'].Plastic(table=((355,0),
                                                    (387.9,0.0006),
                                                    (393.63,0.0009),
                                                    (400.99,0.0023),
                                                    (401.7,0.0056),
                                                    (427.5,0.0106),
                                                    (453.55,0.0154),
                                                    (472.3,0.0201),
                                                    (486.68,0.0246),
                                                    (497.61,0.0288),
                                                    (507.9,0.0342),
                                                    (515.25,0.0389),
                                                    (521.11,0.0437),
                                                    (525.73,0.0482),
                                                    (529.42,0.0529),
                                                    (532.03,0.0573),
                                                    (531.29,0.0622),))
        
    del mdb.models['Model-2'].materials['Steel']
    
    
    
    mdb.models['Model-2'].Material(name='Stiff_support-nonlinear', 
        objectToCopy=mdb.models['Model-2'].materials['Stiff_support'])
    # mdb.models['Model-2'].materials['Stiff_support-nonlinear'].Plastic(table=((
        # 355.0, 0.0), (510.0, 0.2)))
    del mdb.models['Model-2'].materials['Stiff_support']
    

    
    # assign the material to the parts again
    
    mdb.models['Model-2'].sections['tf1'].setValues(preIntegrate=OFF, 
        material='Steel-nonlinear', thicknessType=UNIFORM, thickness=topflangethickness_tf1, 
        thicknessField='', nodalThicknessField='', 
        idealization=NO_IDEALIZATION, integrationRule=SIMPSON, numIntPts=5)
    
    
    
    mdb.models['Model-2'].sections['tw'].setValues(preIntegrate=OFF, 
        material='Steel-nonlinear', thicknessType=UNIFORM, thickness=webthickness_tw, 
        thicknessField='', nodalThicknessField='', 
        idealization=NO_IDEALIZATION, integrationRule=SIMPSON, numIntPts=5)
    
    

    mdb.models['Model-2'].sections['tf2_span'].setValues(preIntegrate=OFF, 
        material='Steel-nonlinear', thicknessType=UNIFORM, thickness=bottomflangethickness_tf2, 
        thicknessField='', nodalThicknessField='', 
        idealization=NO_IDEALIZATION, integrationRule=SIMPSON, numIntPts=5)
    
    
    
    mdb.models['Model-2'].sections['tf2_support'].setValues(preIntegrate=OFF, 
        material='Stiff_support-nonlinear', thicknessType=UNIFORM, 
        thickness=bottomflangethickness_tf2, thicknessField='', nodalThicknessField='', 
        idealization=NO_IDEALIZATION, integrationRule=SIMPSON, numIntPts=5)
    
    
    
    
    
    # delete the previous
  
    del mdb.models['Model-2'].steps['Step-1']
    
    
    
    
    
    # create a new step: static risk
    
    # mdb.models['Model-2'].StaticRiksStep(name='Step-1', previous='Initial', 
        # maxNumInc=5000, initialArcInc=0.01, minArcInc=1e-5, maxArcInc=1e+36, 
        # totalArcLength=1.0, nlgeom=ON)
        
    # mdb.models['Model-2'].StaticRiksStep(name='Step-1', previous='Initial', 
        # maxNumInc=5000, initialArcInc=0.01, nlgeom=ON)
    
    
    # create a new step: static analysis
    
    mdb.models['Model-2'].StaticStep(name='Step-1', previous='Initial', 
        maxNumInc=increments_steps, initialInc=initial_steps, maxInc=max_steps, nlgeom=ON)
    
    
    
    
    ######################all the B.C. remain as they are in the initial step####################
    
    
    
    #################################                   NEW PRESSURE LOAD                  ###################################
    
    # a = mdb.models['Model-2'].rootAssembly
    # s1 = a.instances['I_Section-1'].faces
    # side2Faces1 = s1.getSequenceFromMask(mask=('[#18f03000 #614a6303 #318 ]', ), )
    # region = a.Surface(side2Faces=side2Faces1, name='Surf-1')
    # mdb.models['Model-2'].Pressure(name='UDL_load', createStepName='Step-1', 
        # region=region, distributionType=UNIFORM, field='', magnitude=UDL)
    
    
    
    #################################                   BODY FORCE                  ###################################
    
    a = mdb.models['Model-2'].rootAssembly
    f1 = a.instances['I_Section-1'].faces
    faces1 = f1.getSequenceFromMask(mask=('[#21000556 #80040000 ]', ), )
    region = regionToolset.Region(faces=faces1)
    mdb.models['Model-2'].BodyForce(name='UDL_load', createStepName='Step-1', 
        region=region, comp2=(-UDL/((web_depth/2)*webthickness_tw)))
    
    
    
    
    # open the edit keyword
    
    mdb.models['Model-2'].keywordBlock.synchVersions(storeNodesAndElements=False)
    
    
    
    # add the field
    
    mdb.models['Model-2'].keywordBlock.replace(54, """
** ----------------------------------------------------------------
*IMPERFECTION, FILE=Job-1, STEP=1
1, %.1f
**
** STEP: Step-1
    **""" % scale_factor)
    

    
    ####################
    
    # to request information ------- for now I keep them all to check
    
    del mdb.models['Model-2'].fieldOutputRequests['F-Output-1']
    # mdb.models['Model-2'].fieldOutputRequests['F-Output-2'].setValues(variables=(
        # 'U', 'BF', 'S', 'MISES'))
    
    mdb.models['Model-2'].FieldOutputRequest(name='F-Output-2', 
        createStepName='Step-1', variables=('S', 'MISES', 'U', 'BF'))
    
    
    
    
    
    
    ####################
    
    # EIGEN MODE - NODES SETS -
    
  
    ##############            Displacement points             ################
    
    a = mdb.models['Model-2'].rootAssembly
    n1 = a.instances['I_Section-1'].nodes
    nodes1 = n1.getSequenceFromMask(mask=('[#0:2 #20 ]', ), )
    a.Set(nodes=nodes1, name='Set-1')
    
    a = mdb.models['Model-2'].rootAssembly
    n1 = a.instances['I_Section-1'].nodes
    nodes1 = n1.getSequenceFromMask(mask=('[#0 #200 ]', ), )
    a.Set(nodes=nodes1, name='Set-2')
    
    
    
    ##############            Shear buckling points             ################
    
    a = mdb.models['Model-2'].rootAssembly
    n1 = a.instances['I_Section-1'].nodes
    nodes1 = n1.getSequenceFromMask(mask=('[#100 ]', ), )
    a.Set(nodes=nodes1, name='Set-3')
    
    a = mdb.models['Model-2'].rootAssembly
    n1 = a.instances['I_Section-1'].nodes
    nodes1 = n1.getSequenceFromMask(mask=('[#4000 ]', ), )
    a.Set(nodes=nodes1, name='Set-4')
    
    a = mdb.models['Model-2'].rootAssembly
    n1 = a.instances['I_Section-1'].nodes
    nodes1 = n1.getSequenceFromMask(mask=('[#80000 ]', ), )
    a.Set(nodes=nodes1, name='Set-5')
    
    a = mdb.models['Model-2'].rootAssembly
    n1 = a.instances['I_Section-1'].nodes
    nodes1 = n1.getSequenceFromMask(mask=('[#40 ]', ), )
    a.Set(nodes=nodes1, name='Set-6')
    
    a = mdb.models['Model-2'].rootAssembly
    n1 = a.instances['I_Section-1'].nodes
    nodes1 = n1.getSequenceFromMask(mask=('[#10 ]', ), )
    a.Set(nodes=nodes1, name='Set-7')
    
    a = mdb.models['Model-2'].rootAssembly
    n1 = a.instances['I_Section-1'].nodes
    nodes1 = n1.getSequenceFromMask(mask=('[#1 ]', ), )
    a.Set(nodes=nodes1, name='Set-8')
    
    
    
    #############      Lateral torsional buckling points      ################
    
    a = mdb.models['Model-2'].rootAssembly
    n1 = a.instances['I_Section-1'].nodes
    nodes1 = n1.getSequenceFromMask(mask=('[#0 #10000 ]', ), )
    a.Set(nodes=nodes1, name='Set-9')
    
    a = mdb.models['Model-2'].rootAssembly
    n1 = a.instances['I_Section-1'].nodes
    nodes1 = n1.getSequenceFromMask(mask=('[#0 #800 ]', ), )
    a.Set(nodes=nodes1, name='Set-10')
    
    a = mdb.models['Model-2'].rootAssembly
    n1 = a.instances['I_Section-1'].nodes
    nodes1 = n1.getSequenceFromMask(mask=('[#0 #100000 ]', ), )
    a.Set(nodes=nodes1, name='Set-11')
    
    a = mdb.models['Model-2'].rootAssembly
    n1 = a.instances['I_Section-1'].nodes
    nodes1 = n1.getSequenceFromMask(mask=('[#0 #40000 ]', ), )
    a.Set(nodes=nodes1, name='Set-12')
    
    a = mdb.models['Model-2'].rootAssembly
    n1 = a.instances['I_Section-1'].nodes
    nodes1 = n1.getSequenceFromMask(mask=('[#0 #10 ]', ), )
    a.Set(nodes=nodes1, name='Set-13')
    
    a = mdb.models['Model-2'].rootAssembly
    n1 = a.instances['I_Section-1'].nodes
    nodes1 = n1.getSequenceFromMask(mask=('[#0 #4 ]', ), )
    a.Set(nodes=nodes1, name='Set-14')
    
    
    
    #############      Compression buckling points      ################
    
    a = mdb.models['Model-2'].rootAssembly
    n1 = a.instances['I_Section-1'].nodes
    nodes1 = n1.getSequenceFromMask(mask=('[#0 #200000 ]', ), )
    a.Set(nodes=nodes1, name='Set-15')
    
    a = mdb.models['Model-2'].rootAssembly
    n1 = a.instances['I_Section-1'].nodes
    nodes1 = n1.getSequenceFromMask(mask=('[#0 #4000 ]', ), )
    a.Set(nodes=nodes1, name='Set-16')
    
    a = mdb.models['Model-2'].rootAssembly
    n1 = a.instances['I_Section-1'].nodes
    nodes1 = n1.getSequenceFromMask(mask=('[#0 #8000 ]', ), )
    a.Set(nodes=nodes1, name='Set-17')
    
    a = mdb.models['Model-2'].rootAssembly
    n1 = a.instances['I_Section-1'].nodes
    nodes1 = n1.getSequenceFromMask(mask=('[#0 #1000 ]', ), )
    a.Set(nodes=nodes1, name='Set-18')
    
    a = mdb.models['Model-2'].rootAssembly
    n1 = a.instances['I_Section-1'].nodes
    nodes1 = n1.getSequenceFromMask(mask=('[#0 #400 ]', ), )
    a.Set(nodes=nodes1, name='Set-19')
    
    a = mdb.models['Model-2'].rootAssembly
    n1 = a.instances['I_Section-1'].nodes
    nodes1 = n1.getSequenceFromMask(mask=('[#80000000 ]', ), )
    a.Set(nodes=nodes1, name='Set-20')
    
    a = mdb.models['Model-2'].rootAssembly
    n1 = a.instances['I_Section-1'].nodes
    nodes1 = n1.getSequenceFromMask(mask=('[#10000000 ]', ), )
    a.Set(nodes=nodes1, name='Set-21')
    
    a = mdb.models['Model-2'].rootAssembly
    n1 = a.instances['I_Section-1'].nodes
    nodes1 = n1.getSequenceFromMask(mask=('[#40000000 ]', ), )
    a.Set(nodes=nodes1, name='Set-22')
    
    a = mdb.models['Model-2'].rootAssembly
    n1 = a.instances['I_Section-1'].nodes
    nodes1 = n1.getSequenceFromMask(mask=('[#0 #100 ]', ), )
    a.Set(nodes=nodes1, name='Set-23')
    
    
    
    
    
    
    
    ##################        HISTORY OUTPUT          #################
   
    del mdb.models['Model-2'].historyOutputRequests['H-Output-1']
    
    
    ##############            Displacement points             ################
    
    # Set-1
    regionDef=mdb.models['Model-2'].rootAssembly.sets['Set-1']
    mdb.models['Model-2'].HistoryOutputRequest(name='H-Output-1', 
        createStepName='Step-1', variables=('U1', 'U2', 'U3', 'UR1', 'UR2', 
        'UR3'), region=regionDef, sectionPoints=DEFAULT, rebar=EXCLUDE)
    
    
    # Set-2
    regionDef=mdb.models['Model-2'].rootAssembly.sets['Set-2']
    mdb.models['Model-2'].HistoryOutputRequest(name='H-Output-2', 
        createStepName='Step-1', variables=('U1', 'U2', 'U3', 'UR1', 'UR2', 
        'UR3'), region=regionDef, sectionPoints=DEFAULT, rebar=EXCLUDE)
    
    
    # Set-3
    regionDef=mdb.models['Model-2'].rootAssembly.sets['Set-3']
    mdb.models['Model-2'].HistoryOutputRequest(name='H-Output-3', 
        createStepName='Step-1', variables=('U1', 'U2', 'U3', 'UR1', 'UR2', 
        'UR3'), region=regionDef, sectionPoints=DEFAULT, rebar=EXCLUDE)
    
    
    # Set-4
    regionDef=mdb.models['Model-2'].rootAssembly.sets['Set-4']
    mdb.models['Model-2'].HistoryOutputRequest(name='H-Output-4', 
        createStepName='Step-1', variables=('U1', 'U2', 'U3', 'UR1', 'UR2', 
        'UR3'), region=regionDef, sectionPoints=DEFAULT, rebar=EXCLUDE)
    
    
    # Set-5
    regionDef=mdb.models['Model-2'].rootAssembly.sets['Set-5']
    mdb.models['Model-2'].HistoryOutputRequest(name='H-Output-5', 
        createStepName='Step-1', variables=('U1', 'U2', 'U3', 'UR1', 'UR2', 
        'UR3'), region=regionDef, sectionPoints=DEFAULT, rebar=EXCLUDE)
    
    
    # Set-6
    regionDef=mdb.models['Model-2'].rootAssembly.sets['Set-6']
    mdb.models['Model-2'].HistoryOutputRequest(name='H-Output-6', 
        createStepName='Step-1', variables=('U1', 'U2', 'U3', 'UR1', 'UR2', 
        'UR3'), region=regionDef, sectionPoints=DEFAULT, rebar=EXCLUDE)
    
    
    # Set-7
    regionDef=mdb.models['Model-2'].rootAssembly.sets['Set-7']
    mdb.models['Model-2'].HistoryOutputRequest(name='H-Output-7', 
        createStepName='Step-1', variables=('U1', 'U2', 'U3', 'UR1', 'UR2', 
        'UR3'), region=regionDef, sectionPoints=DEFAULT, rebar=EXCLUDE)
    
    
    # Set-8
    regionDef=mdb.models['Model-2'].rootAssembly.sets['Set-8']
    mdb.models['Model-2'].HistoryOutputRequest(name='H-Output-8', 
        createStepName='Step-1', variables=('U1', 'U2', 'U3', 'UR1', 'UR2', 
        'UR3'), region=regionDef, sectionPoints=DEFAULT, rebar=EXCLUDE)
    
    
    # Set-9
    regionDef=mdb.models['Model-2'].rootAssembly.sets['Set-9']
    mdb.models['Model-2'].HistoryOutputRequest(name='H-Output-9', 
        createStepName='Step-1', variables=('U1', 'U2', 'U3', 'UR1', 'UR2', 
        'UR3'), region=regionDef, sectionPoints=DEFAULT, rebar=EXCLUDE)
    
    
    # Set-10
    regionDef=mdb.models['Model-2'].rootAssembly.sets['Set-10']
    mdb.models['Model-2'].HistoryOutputRequest(name='H-Output-10', 
        createStepName='Step-1', variables=('U1', 'U2', 'U3', 'UR1', 'UR2', 
        'UR3'), region=regionDef, sectionPoints=DEFAULT, rebar=EXCLUDE)
    
    
    # Set-11
    regionDef=mdb.models['Model-2'].rootAssembly.sets['Set-11']
    mdb.models['Model-2'].HistoryOutputRequest(name='H-Output-11', 
        createStepName='Step-1', variables=('U1', 'U2', 'U3', 'UR1', 'UR2', 
        'UR3'), region=regionDef, sectionPoints=DEFAULT, rebar=EXCLUDE)
    
    
    # Set-12
    regionDef=mdb.models['Model-2'].rootAssembly.sets['Set-12']
    mdb.models['Model-2'].HistoryOutputRequest(name='H-Output-12', 
        createStepName='Step-1', variables=('U1', 'U2', 'U3', 'UR1', 'UR2', 
        'UR3'), region=regionDef, sectionPoints=DEFAULT, rebar=EXCLUDE)
    
    
    # Set-13
    regionDef=mdb.models['Model-2'].rootAssembly.sets['Set-13']
    mdb.models['Model-2'].HistoryOutputRequest(name='H-Output-13', 
        createStepName='Step-1', variables=('U1', 'U2', 'U3', 'UR1', 'UR2', 
        'UR3'), region=regionDef, sectionPoints=DEFAULT, rebar=EXCLUDE)
    
    
    # Set-14
    regionDef=mdb.models['Model-2'].rootAssembly.sets['Set-14']
    mdb.models['Model-2'].HistoryOutputRequest(name='H-Output-14', 
        createStepName='Step-1', variables=('U1', 'U2', 'U3', 'UR1', 'UR2', 
        'UR3'), region=regionDef, sectionPoints=DEFAULT, rebar=EXCLUDE)
    
    
    # Set-15
    regionDef=mdb.models['Model-2'].rootAssembly.sets['Set-15']
    mdb.models['Model-2'].HistoryOutputRequest(name='H-Output-15', 
        createStepName='Step-1', variables=('U1', 'U2', 'U3', 'UR1', 'UR2', 
        'UR3'), region=regionDef, sectionPoints=DEFAULT, rebar=EXCLUDE)
    
    
    # Set-16
    regionDef=mdb.models['Model-2'].rootAssembly.sets['Set-16']
    mdb.models['Model-2'].HistoryOutputRequest(name='H-Output-16', 
        createStepName='Step-1', variables=('U1', 'U2', 'U3', 'UR1', 'UR2', 
        'UR3'), region=regionDef, sectionPoints=DEFAULT, rebar=EXCLUDE)
    
    
    # Set-17
    regionDef=mdb.models['Model-2'].rootAssembly.sets['Set-17']
    mdb.models['Model-2'].HistoryOutputRequest(name='H-Output-17', 
        createStepName='Step-1', variables=('U1', 'U2', 'U3', 'UR1', 'UR2', 
        'UR3'), region=regionDef, sectionPoints=DEFAULT, rebar=EXCLUDE)
    
    
    # Set-18
    regionDef=mdb.models['Model-2'].rootAssembly.sets['Set-18']
    mdb.models['Model-2'].HistoryOutputRequest(name='H-Output-18', 
        createStepName='Step-1', variables=('U1', 'U2', 'U3', 'UR1', 'UR2', 
        'UR3'), region=regionDef, sectionPoints=DEFAULT, rebar=EXCLUDE)
    
    
    # Set-19
    regionDef=mdb.models['Model-2'].rootAssembly.sets['Set-19']
    mdb.models['Model-2'].HistoryOutputRequest(name='H-Output-19', 
        createStepName='Step-1', variables=('U1', 'U2', 'U3', 'UR1', 'UR2', 
        'UR3'), region=regionDef, sectionPoints=DEFAULT, rebar=EXCLUDE)
    
    
    # Set-20
    regionDef=mdb.models['Model-2'].rootAssembly.sets['Set-20']
    mdb.models['Model-2'].HistoryOutputRequest(name='H-Output-20', 
        createStepName='Step-1', variables=('U1', 'U2', 'U3', 'UR1', 'UR2', 
        'UR3'), region=regionDef, sectionPoints=DEFAULT, rebar=EXCLUDE)
    
    
    # Set-21
    regionDef=mdb.models['Model-2'].rootAssembly.sets['Set-21']
    mdb.models['Model-2'].HistoryOutputRequest(name='H-Output-21', 
        createStepName='Step-1', variables=('U1', 'U2', 'U3', 'UR1', 'UR2', 
        'UR3'), region=regionDef, sectionPoints=DEFAULT, rebar=EXCLUDE)
    
    
    # Set-22
    regionDef=mdb.models['Model-2'].rootAssembly.sets['Set-22']
    mdb.models['Model-2'].HistoryOutputRequest(name='H-Output-22', 
        createStepName='Step-1', variables=('U1', 'U2', 'U3', 'UR1', 'UR2', 
        'UR3'), region=regionDef, sectionPoints=DEFAULT, rebar=EXCLUDE)
    
    
    # Set-23
    regionDef=mdb.models['Model-2'].rootAssembly.sets['Set-23']
    mdb.models['Model-2'].HistoryOutputRequest(name='H-Output-23', 
        createStepName='Step-1', variables=('U1', 'U2', 'U3', 'UR1', 'UR2', 
        'UR3'), region=regionDef, sectionPoints=DEFAULT, rebar=EXCLUDE)
    
    
    
    
    
    
    
    # create the new job
    
    # mdb.Job(name='Job-2', model='Model-2', description='', type=ANALYSIS, 
        # atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90, 
        # memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True, 
        # explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, echoPrint=OFF, 
        # modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine='', 
        # scratch='', resultsFormat=ODB, multiprocessingMode=DEFAULT, numCpus=1, 
        # numGPUs=0)
    
    mdb.Job(name='Job-2', model='Model-2', description='', type=ANALYSIS, 
        atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90, 
        memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True, 
        explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, echoPrint=OFF, 
        modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine='', 
        scratch='', resultsFormat=ODB, parallelizationMethodExplicit=DOMAIN, 
        numDomains=3, activateLoadBalancing=False, multiprocessingMode=DEFAULT, 
        numCpus=3, numGPUs=0)
    
    
    
    # SUBMIT THE JOB

    mdb.jobs['Job-2'].submit(consistencyChecking=OFF)
    
    
    # TO WAIT FOR JOB COMPLETION
    
    mdb.jobs['Job-2'].waitForCompletion()
    print("Job-2 finished running")
    

    
    ####################
    
    # FAILURE MODE INFO -
    
    
    ########################       1st open odb (output database) in original file        #############################
    
    
    odb = session.openOdb(str(BeamFolder)+'/'+'Job-2.odb')
    
    
    
    #'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''#
    
    ##############            MAX DISPLACEMENT IN ALL DIRECTIONS             ################
    
    ############################## Value at this node  ###########################

    NrOfSteps = len(odb.steps['Step-1'].frames)
    Forces = []
    Displacements = []
    Displacements2 = []
    Displacements3 = []
    Yield_Disp = []
    Stiffness = []
    
    ############################## SAVE DATA IN EXCEL FILE ###########################
    
    NewFolder = BeamFolder+"/"+str("MAX_DISPLACMENT")+"_"+str("U1_U2_U3")
    os.mkdir(NewFolder)
    os.chdir(NewFolder)
    
    opFile = NewFolder+'/'+'MAX_DISPLACMENT_Job-2'+'-ForceDispCurve.csv'
    
    try:
        opFileU = open(opFile,'w')
        opFileU.write("%10s,%10s,%10s,%10s,%10s,%10s\n"%('Force', 'U1_Max', 'U2_Max', 'U3_Max', 'Yield_Disp', 'Stiffness') )
    except IOError:
        print 'cannot open', opFile
        exit(0)

    for i in range(NrOfSteps):
        Disps = odb.steps['Step-1'].frames[i].fieldOutputs['U'].getSubset(position = NODAL).bulkDataBlocks[0].data
        MaxDisps = np.max(np.abs(Disps),axis=0)
        Displacements.append([MaxDisps[0]])
        Displacements2.append([MaxDisps[1]])
        Displacements3.append([MaxDisps[2]])
        Yielding_displacement = (MaxDisps[0]**2.0 + MaxDisps[1]**2.0 + MaxDisps[2]**2.0)**(0.5)
        Yield_Disp.append([Yielding_displacement])
        Applied_Load = odb.steps['Step-1'].frames[i].fieldOutputs['BF'].values[0].data[1]*((web_depth/2)*webthickness_tw)*(-1.0)
        Forces.append(Applied_Load)
        if i!=0 and Yield_Disp[i-1][0]!=Yield_Disp[i][0]:
            Current_Stiffness = (Forces[i]-Forces[i-1])/(Yield_Disp[i][0]-Yield_Disp[i-1][0])
            Stiffness.append([Current_Stiffness])
        else:
            Current_Stiffness = 999999.0
            Stiffness.append([Current_Stiffness])
        opFileU.write("%10f,%10f,%10f,%10f,%10f,%10f\n" % (Applied_Load, MaxDisps[0], MaxDisps[1], MaxDisps[2], Yielding_displacement, Current_Stiffness))
        
    opFileU.close()
    
    
    ####################### MAKE A PLOT AND SAVE ######################
   
    fig, ax = plt.subplots()
    # ax.plot(Displacements, Displacements2, Displacements3, Forces)
    # Plotting both the curves simultaneously
    ax.plot(Displacements, Forces, color='r', label='U1_MAX')
    ax.plot(Displacements2, Forces, color='g', label='U2_Max')
    ax.plot(Displacements3, Forces, color='b', label='U3_Max')
    ax.plot(Yield_Disp, Forces, color='y', label='Yield_displacement')
    
    plt.legend()
    

    ax.set(xlabel='Displacements (mm)', ylabel='Linear Load (N/mm)',
           title='Force Displacement Curve')
    ax.grid()

    fig.savefig("MAX_DISPLACMENT.png")
    plt.close(fig)
    ####################################################################
    
    
    
    #'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''#
    
    ##############            Displacement             ################
    
    NewFolderprincipal_1 = BeamFolder+"/"+str("Vertical-Displacement")
    os.mkdir(NewFolderprincipal_1)
    os.chdir(NewFolderprincipal_1)
    
    node_table = [["70","U2"],
                  ["41","U2"],
                  ["42","U2"]]
    
    # This creates a list full of zeros to combine the displacements
    VerticalDisp = []
    for i in range(NrOfSteps):
        VerticalDisp.append([0.0, 0.0])
    
    
    #This prints all the displacements and squares them inside the combined list
    for i in node_table:
        bbbbbbb = odb.steps['Step-1'].historyRegions['Node I_SECTION-1.%s'%i[0]].historyOutputs['%s'%i[1]].data
        for j in range(NrOfSteps):
            VerticalDisp[j][1] = VerticalDisp[j][1] + bbbbbbb[j][1]**2.0
        SaveInExcel(NewFolderprincipal_1, 'Node_%s_%s_Job-2' % (i[0],i[1]), bbbbbbb, Forces)
        MakePlot("Plot_Node_%s_%s_Job-2" % (i[0],i[1]), bbbbbbb, Forces)

    
    # This square roots the combined displacements
    for j in range(NrOfSteps):
        VerticalDisp[j][1] = VerticalDisp[j][1]**(0.5)
    
    SaveInExcel(NewFolderprincipal_1, 'Combined_VertDisp', VerticalDisp, Forces)
    MakePlot('Plot_Combined_VertDisp', VerticalDisp, Forces)
    
    
    #'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''#
    
    ##############            Shear buckling             ################
    
    NewFolderprincipal_2 = BeamFolder+"/"+str("Shear-Buckling")
    os.mkdir(NewFolderprincipal_2)
    os.chdir(NewFolderprincipal_2)
    
    
    node_table = [["9","U3"],
                  ["15","U3"],
                  ["20","U3"],
                  ["7","U3"],
                  ["5","U3"],
                  ["1","U3"]]
    
    # This creates a list full of zeros to combine the displacements
    ShearBucklingDisp = []
    for i in range(NrOfSteps):
        ShearBucklingDisp.append([0.0, 0.0])
        
        
    #This prints all the displacements and squares them inside the combined list
    for i in node_table:
        bbbbbbb = odb.steps['Step-1'].historyRegions['Node I_SECTION-1.%s'%i[0]].historyOutputs['%s'%i[1]].data
        for j in range(NrOfSteps):
            ShearBucklingDisp[j][1] = max(np.abs(ShearBucklingDisp[j][1]), np.abs(bbbbbbb[j][1]))
        SaveInExcel(NewFolderprincipal_2, 'Node_%s_%s_Job-2' % (i[0],i[1]), bbbbbbb, Forces)
        MakePlot("Plot_Node_%s_%s_Job-2" % (i[0],i[1]), bbbbbbb, Forces)

    SaveInExcel(NewFolderprincipal_2, 'Combined_ShearBucklingDisp', ShearBucklingDisp, Forces)
    MakePlot('Plot_Combined_ShearBucklingDisp', ShearBucklingDisp, Forces)
        
        
    
    # #This prints all the displacements and squares them inside the combined list
    # for i in node_table:
        # bbbbbbb = odb.steps['Step-1'].historyRegions['Node I_SECTION-1.%s'%i[0]].historyOutputs['%s'%i[1]].data
        # for j in range(NrOfSteps):
            # ShearBucklingDisp[j][1] = ShearBucklingDisp[j][1] + bbbbbbb[j][1]**2.0
        # SaveInExcel(NewFolderprincipal_2, 'Node_%s_%s_Job-2' % (i[0],i[1]), bbbbbbb, Forces)
        # MakePlot("Plot_Node_%s_%s_Job-2" % (i[0],i[1]), bbbbbbb, Forces)


    # # This square roots the combined displacements
    # # ShearBucklingStiffness = []
    # for j in range(NrOfSteps):
        # ShearBucklingDisp[j][1] = ShearBucklingDisp[j][1]**(0.5)/len(node_table)
        # # if j!=0 and ShearBucklingDisp[j-1][1]!=ShearBucklingDisp[j][1]:
            # # Current_Stiffness = (Forces[j]-Forces[j-1])/(ShearBucklingDisp[j][1]-ShearBucklingDisp[j-1][1])
            # # ShearBucklingStiffness.append([Current_Stiffness])
        # # else:
            # # Current_Stiffness = 999999.0
            # # ShearBucklingStiffness.append([Current_Stiffness])
    
    # SaveInExcel(NewFolderprincipal_2, 'Combined_ShearBucklingDisp', ShearBucklingDisp, Forces)
    # MakePlot('Plot_Combined_ShearBucklingDisp', ShearBucklingDisp, Forces)
    
    
    #############      Lateral torsional buckling      ################
    
    NewFolderprincipal_3 = BeamFolder+"/"+str("Lateral-Torsional-Buckling")
    os.mkdir(NewFolderprincipal_3)
    os.chdir(NewFolderprincipal_3)
    
    
    node_table = [["70","U3"],
                  ["42","U3"],
                  ["49","U3"],
                  ["44","U3"],
                  ["53","U3"],
                  ["51","U3"],
                  ["37","U3"],
                  ["35","U3"],
                  ["49","U2"],
                  ["44","U2"],
                  ["53","U2"],
                  ["51","U2"],
                  ["37","U2"],
                  ["35","U2"]]
    
    # This creates a list full of zeros to combine the displacements
    LTBucklingDisp = []
    for i in range(NrOfSteps):
        LTBucklingDisp.append([0.0, 0.0])
    
    
    #This prints all the displacements and squares them inside the combined list
    for i in node_table:
        bbbbbbb = odb.steps['Step-1'].historyRegions['Node I_SECTION-1.%s'%i[0]].historyOutputs['%s'%i[1]].data
        for j in range(NrOfSteps):
            LTBucklingDisp[j][1] = max(np.abs(LTBucklingDisp[j][1]), np.abs(bbbbbbb[j][1]))
        SaveInExcel(NewFolderprincipal_3, 'Node_%s_%s_Job-2' % (i[0],i[1]), bbbbbbb, Forces)
        MakePlot("Plot_Node_%s_%s_Job-2" % (i[0],i[1]), bbbbbbb, Forces)


    # This square roots the combined displacements
    # LTBucklingStiffness = []
    # for j in range(NrOfSteps):
        # LTBucklingDisp[j][1] = LTBucklingDisp[j][1]**(0.5)
        # if j!=0 and LTBucklingDisp[j-1][1]!=LTBucklingDisp[j][1]:
            # Current_Stiffness = (Forces[j]-Forces[j-1])/(LTBucklingDisp[j][1]-LTBucklingDisp[j-1][1])
            # LTBucklingStiffness.append([Current_Stiffness])
        # else:
            # Current_Stiffness = 999999.0
            # LTBucklingStiffness.append([Current_Stiffness])
    
    SaveInExcel(NewFolderprincipal_3, 'Combined_LTBucklingDisp', LTBucklingDisp, Forces)
    MakePlot('Plot_Combined_LTBucklingDisp', LTBucklingDisp, Forces)
    
    
    
    #############      Compression buckling      ################
    
    NewFolderprincipal_4 = BeamFolder+"/"+str("Compression-Buckling")
    os.mkdir(NewFolderprincipal_4)
    os.chdir(NewFolderprincipal_4)
    
    
    node_table = [["54","U2"],
                  ["47","U2"],
                  ["48","U2"],
                  ["45","U2"],
                  ["49","U2"],
                  ["44","U2"],
                  ["43","U2"],
                  ["32","U2"],
                  ["29","U2"],
                  ["31","U2"]]
    
    # This creates a list full of zeros to combine the displacements
    CompBucklingDisp = []
    for i in range(NrOfSteps):
        CompBucklingDisp.append([0.0, 0.0])
    
    
    #This prints all the displacements and squares them inside the combined list
    for i in node_table:
        bbbbbbb = odb.steps['Step-1'].historyRegions['Node I_SECTION-1.%s'%i[0]].historyOutputs['%s'%i[1]].data
        for j in range(NrOfSteps):
            CompBucklingDisp[j][1] = max(np.abs(CompBucklingDisp[j][1]), np.abs(bbbbbbb[j][1]))
        SaveInExcel(NewFolderprincipal_4, 'Node_%s_%s_Job-2' % (i[0],i[1]), bbbbbbb, Forces)
        MakePlot("Plot_Node_%s_%s_Job-2" % (i[0],i[1]), bbbbbbb, Forces)


    # This square roots the combined displacements
    # CompBucklingStiffness = []
    # for j in range(NrOfSteps):
        # CompBucklingDisp[j][1] = CompBucklingDisp[j][1]**(0.5)/len(node_table)
        # if j!=0 and CompBucklingDisp[j-1][1]!=CompBucklingDisp[j][1]:
            # Current_Stiffness = (Forces[j]-Forces[j-1])/(CompBucklingDisp[j][1]-CompBucklingDisp[j-1][1])
            # CompBucklingStiffness.append([Current_Stiffness])
        # else:
            # Current_Stiffness = 999999.0
            # CompBucklingStiffness.append([Current_Stiffness])
    
    SaveInExcel(NewFolderprincipal_4, 'Combined_CompBucklingDisp', CompBucklingDisp, Forces)
    MakePlot('Plot_Combined_CompBucklingDisp', CompBucklingDisp, Forces)
    
    ######################################################################################################
    
    odb.close()
    
    Weight = web_depth*webthickness_tw + topflange_b1*topflangethickness_tf1 + bottomflange_b2*bottomflangethickness_tf2
    
    InitialStiffness = Stiffness[1][0]
    Stiffness_ratio = 0.19        # THIS IS TO MAKE SURE WE HAVE A MIN NUMBER, SO NO ISSUES OCCUR
    for i in range(NrOfSteps):
        if Forces[i]>=Design_load:
            Stiffness_ratio = Stiffness[i][0]/Stiffness[1][0]
            break
    
    # Yield_load = np.max(Forces)
    # for i in range(NrOfSteps):
        # if Stiffness[i][0]<InitialStiffness*0.7:
            # Yield_load = Forces[i]
            # break
    
    # DO NOT FORGET TO ADD WEIGHT_AVERAGE THINGY TO PENALTIES
    Penalty_yielding = weight_average*(Stiffness_ratio < 0.6)*(1.0/Stiffness_ratio)/5.0
    Penalty_yielding = Penalty_yielding + weight_average*(Stiffness_ratio < 0.4)*(1.0/Stiffness_ratio)/2.0
    Penalty_yielding = Penalty_yielding + weight_average*(Stiffness_ratio < 0.2)*(1.0/Stiffness_ratio)
    
    Max_load = np.max(Forces)
    Penalty_MaxLoad = weight_average*(Max_load < Design_load)*(Design_load/Max_load)/2.0
    Penalty_MaxLoad = Penalty_MaxLoad + weight_average*(Max_load < Accidental_load)*(Accidental_load/Max_load)/2.0
    
    
    
    # disp_limit = beam_span/360.0
    # disp_SLSLoad = 0.0
    # for i in range(NrOfSteps):
        # if Forces[i]>Serviceability_load:
            # disp_SLSLoad = VerticalDisp[i][1]
            # break   
    
    # Penalty_VerticalDisp = weight_average*(disp_SLSLoad > disp_limit)*(disp_SLSLoad/disp_limit)
    
    disp_limit = beam_span/360.0
    disp_SLSLoad = np.max(Forces)
    for j in range(NrOfSteps):
        if VerticalDisp[j][1] > disp_limit:
            disp_SLSLoad = Forces[j]
            break
    
    Penalty_VerticalDisp = weight_average*(disp_SLSLoad < Serviceability_load)*(Serviceability_load/disp_SLSLoad)
    
    
    disp_limit_shear = web_depth/360.0
    # InitialStiffness = ShearBucklingStiffness[1][0]
    ShearBuckling_load = np.max(Forces)
    for j in range(NrOfSteps):
        if ShearBucklingDisp[j][1] > disp_limit_shear:
            ShearBuckling_load = Forces[j]
            break
    # for i in range(NrOfSteps):
        # if ShearBucklingStiffness[i][0]<InitialStiffness*0.2:
            # ShearBuckling_load = Forces[i]
            # break
    
    
    Penalty_ShearBuckling = weight_average*(ShearBuckling_load < Design_load)*(Design_load/ShearBuckling_load)/10.0
    
    # InitialStiffness = LTBucklingStiffness[1][0]
    # LTBuckling_load = np.max(Forces)
    # for i in range(NrOfSteps):
        # if LTBucklingStiffness[i][0]<InitialStiffness*0.2:
            # LTBuckling_load = Forces[i]
            # break
    
    disp_limit_LT = beam_span/200.0
    LTBuckling_load = np.max(Forces)
    for j in range(NrOfSteps):
        if LTBucklingDisp[j][1] > disp_limit_LT:
            LTBuckling_load = Forces[j]
            break 
    
    Penalty_LTBuckling = weight_average*(LTBuckling_load < Design_load)*(Design_load/LTBuckling_load)/10.0
    
    # InitialStiffness = CompBucklingStiffness[1][0]
    # CompBuckling_load = np.max(Forces)
    # for i in range(NrOfSteps):
        # if CompBucklingStiffness[i][0]<InitialStiffness*0.2:
            # CompBuckling_load = Forces[i]
            # break
    
    disp_limit_Comp = beam_span/200.0
    CompBuckling_load = np.max(Forces)
    for j in range(NrOfSteps):
        if CompBucklingDisp[j][1] > disp_limit_Comp:
            CompBuckling_load = Forces[j]
            break 
    
    
    Penalty_CompBuckling = weight_average*(CompBuckling_load < Design_load)*(Design_load/CompBuckling_load)/10.0
    
    # max(Penalty_yielding, Penalty_MaxLoad, Penalty_VerticalDisp, Penalty_ShearBuckling, Penalty_LTBuckling, Penalty_CompBuckling)
    Fitness = Weight + Penalty_yielding + Penalty_MaxLoad + Penalty_VerticalDisp + Penalty_ShearBuckling + Penalty_LTBuckling + Penalty_CompBuckling
    

    opFile = BeamFolder+"/"+'FitnessData'+'.txt'
    
    try:
        opFileU = open(opFile,'w')
    except IOError:
        print('cannot open', opFile)
        exit(0)

    opFileU.write('the model runned in %s steps\n' % NrOfSteps)
    opFileU.write('\n\nWeight\n')
    opFileU.write(str(Weight))
    opFileU.write('\n\nStiffness_ratio\n')
    if Stiffness_ratio > 0.6:
        opFileU.write("No Yielding occured")
    else:
        opFileU.write(str(Stiffness_ratio))
    opFileU.write('\n\nPenalty_yielding\n')
    opFileU.write(str(Penalty_yielding))
    opFileU.write('\n\nMax_Load\n')
    opFileU.write(str(Max_load))
    opFileU.write('\n\nPenalty_MaxLoad\n')
    opFileU.write(str(Penalty_MaxLoad))
    opFileU.write('\n\ndisp_SLSLoad\tServiceability_Load\n')
    opFileU.write("%10s\t%10s\n"%(str(disp_SLSLoad),str(Serviceability_load)))
    opFileU.write('\n\nPenalty_VerticalDisp\n')
    opFileU.write(str(Penalty_VerticalDisp))
    opFileU.write('\n\nPenalty_ShearBuckling\n')
    opFileU.write(str(Penalty_ShearBuckling))
    opFileU.write('\n\nPenalty_LTBuckling\n')
    opFileU.write(str(Penalty_LTBuckling))
    opFileU.write('\n\nPenalty_CompBuckling\n')
    opFileU.write(str(Penalty_CompBuckling))
    opFileU.write('\n\nFitness\n')
    opFileU.write(str(Fitness))
    opFileU.close()
    
    print("A model just ran")
    return Weight, Fitness
    

def PrintData(Folder, Filename, Title, Data):
    #This will depend on the application in mind
    opFile = Folder+"/"+Filename+'.txt'
    
    try:
        opFileU = open(opFile,'w')
        opFileU.write("%10s\n" % Title)
    except IOError:
        print('cannot open', opFile)
        exit(0)

    for Chromossome in Data:
        opFileU.write(str(Chromossome))
        opFileU.write("\n")
    opFileU.close()


def PrintFinalData(Folder, Filename, Title, Data):
    #This will depend on the application in mind
    opFile = Folder+"/"+Filename+'.txt'
    
    try:
        opFileU = open(opFile,'w')
        opFileU.write("%10s\n" % Title)
    except IOError:
        print('cannot open', opFile)
        exit(0)

    GenCodes = list(Data.keys())
    Fitnesses = list(Data.values())
    for i in range(len(Data)):
        opFileU.write(GenCodes[i])
        opFileU.write("\t")
        opFileU.write(str(Fitnesses[i]))
        # opFileU.write(Chromossome[str(Chromossome[0])])
        opFileU.write("\n")
    opFileU.close()


def DeleteRounds(path):
    os.chdir(path)
    for dirs in os.listdir(path):
        if dirs.startswith("Round"):
            shutil.rmtree("%s/%s" % (path,dirs))

GeneticAlgorithm(variables, Generations, Population_size, Elitism_size, Tournament_size, Round, GA_status)

os.chdir(path)
CreateFolder("TrainingDataforNN")
for i in range(1, Generations+1):
    for dirs in os.listdir("%s/Round%s" % (path,i)):
        if dirs.startswith("["):
            CreateFolder("%s/TrainingDataforNN/%s" % (path,dirs))
            Destiny = "%s/TrainingDataforNN/%s/MAX_DISPLACMENT_Job-2-ForceDispCurve.csv" % (path,dirs)
            Source = "%s/Round%s/%s/MAX_DISPLACMENT_U1_U2_U3/MAX_DISPLACMENT_Job-2-ForceDispCurve.csv" % (path,i,dirs)
            shutil.copyfile(Source, Destiny)
            
PrintToScreen('Training Data is done as well')