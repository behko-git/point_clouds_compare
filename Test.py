import os
import numpy as np
import SFM.Main as Main
from Sensor.utils import read_camera
from Sensor.preprocess import Compose, BGR2RGB    
from Sensor import CursorBasedVideoReader, DataFrameNavSignals
from SFM.Process_Functions import Error_Calculation,Result_Viewer,Save_Results,Cloud_Viewer,Plot_Viewer
import SFM.dem as dem


class Test:
    def __init__(self, Pars_Dict: dict):
        self.Pars_Dict = Pars_Dict
        self.camera_config = read_camera(os.path.join(self.Pars_Dict['signals_path'],'ISAW.txt'))
        self.camera_config = {'cammtx': self.camera_config[0], 'distmtx': self.camera_config[1], 
                              'optcammtx': self.camera_config[2], 'distortroi': self.camera_config[3], 
                              'fov': self.camera_config[4], 'fovx': self.camera_config[5], 
                              'fovy': self.camera_config[6]
                            }
        if self.Pars_Dict['Run_Args']['Number_Of_Images'] != 'auto':
            self.number_of_images = self.Pars_Dict['Run_Args']['Number_Of_Images']

                #* Initial Intrinsic Matrix:
        self.K = self.camera_config['cammtx']

        #* Down Scaling Parameter:
        self.Scale_Comparison=self.Pars_Dict['Run_Args']['Scaling_System']

        #* Down Scaling System:
        self.K[0,0]=self.K[0,0]/self.Scale_Comparison
        self.K[1,1]=self.K[1,1]/self.Scale_Comparison
        self.K[0,2]=self.K[0,2]/self.Scale_Comparison
        self.K[1,2]=self.K[1,2]/self.Scale_Comparison    

        #* Initial First Position:
        self.Initial = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])

        transform = Compose([BGR2RGB()])
        self.video_reader = CursorBasedVideoReader('offline-video', self.Pars_Dict['video_path'], buffer=True, 
                                      video_start_second=Pars_Dict['video_start_second'],
                                      video_slip_ms=Pars_Dict['video_slip_ms'], 
                                      cursor_step=1000, transform=transform)
        Ret, self.img1, self.img2 = self.video_reader.read(increase_cursor=True)

        self.Nav_Sensors = DataFrameNavSignals(
            gps_path=os.path.join(self.Pars_Dict['signals_path'], self.Pars_Dict['signals']['GPS']),
            ref_center_path=os.path.join(self.Pars_Dict['signals_path'], self.Pars_Dict['signals']['ref_center']),
            heading_path=os.path.join(self.Pars_Dict['signals_path'], self.Pars_Dict['signals']['heading']),
            attitude_path=os.path.join(self.Pars_Dict['signals_path'], self.Pars_Dict['signals']['attitude']),
            altimeter_path=os.path.join(self.Pars_Dict['signals_path'], self.Pars_Dict['signals']['height']),
            air_speed_path=os.path.join(self.Pars_Dict['signals_path'], self.Pars_Dict['signals']['speed']),
            barometer_path=os.path.join(self.Pars_Dict['signals_path'], self.Pars_Dict['signals']['barometer']),
            start_index=self.Pars_Dict['video_start_second']+self.Pars_Dict['csv_offset_from_video'],
            stop_index=self.Pars_Dict['video_start_second']+self.Pars_Dict['csv_offset_from_video']+self.Pars_Dict['video_duration_second'],
            step=1, buffer_size=self.Pars_Dict['signals']['buffer_size'],
        )

        self.dem = dem.Dem(r'F:\Data\DEM_map\DemMap_47_part1.tif')
        
        print("Run :")
        print("Processor : "+self.Pars_Dict['Run_Args']['Processing_System'])
        print("Descriptor Algorithm : "+self.Pars_Dict['Run_Args']['Descriptors_Algorithm'])        
        print("Scaling Method : "+self.Pars_Dict['Point_Cloud_Scaling']['Calc_Type'])
        print('---------------------------------------')

    def Run_SFM(self):
        #* Initial Basic Parameters:
        Image_Buffer=None
        Points_Buffer=None

        RMSE_Meter=0
        RMSE_Percent=0
        Mean_RMSE_Meter=0
        Mean_RMSE_Percent=0
        
        Mean_Time=[]
        Mean_Points=[]

        #* Create Objects:
        SFM=Main.SFM(self.Pars_Dict)
        Calc_Error=Error_Calculation(self.Pars_Dict)        
        View_Results=Result_Viewer(self.Pars_Dict)
        Save_Result=Save_Results(self.Pars_Dict)
        View_Cloud_Points=Cloud_Viewer(self.Pars_Dict)

        #* Run Loop :
        for Cloud_Offset in range(self.Pars_Dict['video_start_second'],self.Pars_Dict['video_duration_second']):
        
            #* Read 2 Pictures:
            Ret, image_1, image_2 = self.video_reader.read(increase_cursor=True)

            #* Main SFM run:
            Cloud,Image_Buffer,Points_Buffer,X,Y,Timer_Create=SFM.Create_Point_Cloud(image_1,image_2,self.K,self.Initial,self.Scale_Comparison,Image_Buffer,Points_Buffer)

            #* Initial Real Time Data For Calc Error:
            Sensors = self.Nav_Sensors.read()

            #* Scale SFM:
            Cloud,Timer_Scale=SFM.Scale_Point_Cloud(Cloud,Sensors,self.Scale_Comparison,self.img2.shape[1],X,Y)

            #* Save Point Cloud:
            Save_Result.Save_Cloud(Cloud_Offset,Cloud)

            #* Estimate Number Of Points For Result:
            Mean_Points.append(Cloud.number_of_cells)

            #* Estimate SFM Execute Time:
            Estimated_Time=round(Timer_Create+Timer_Scale,2)
            Mean_Time.append(Estimated_Time)

            cloud = dem.Cloud(self.Pars_Dict['Save_SFM_Path']+'Cloud_'+str(Cloud_Offset)+'.ply')

            compare = dem.Compare(self.dem, cloud)
            compare.plot_bound(fovx=69, fovy=39, sensor_data=Sensors)
                             
            #* Estimate SFM Error:
            if(self.Pars_Dict["Error_Calculation"]["Claculate_Error"]==True):
                RMSE_Percent,RMSE_Meter,Mean_RMSE_Percent,Mean_RMSE_Meter=Calc_Error.Calculate_RMSE(Cloud_Offset,Sensors['altimeter'])

            #* Write CSV Result:
            if(self.Pars_Dict["Results"]["Save_Results_CSV"]==True):
                Save_Result.Save_CSV(Cloud_Offset,Cloud.number_of_cells,Estimated_Time,RMSE_Meter,RMSE_Percent)

            #*Show Created Cloud:  
            if(self.Pars_Dict["Cloud_Viewer"]["View"]==True):
                View_Cloud_Points.View(Cloud_Offset)

            #* Single Result For Each 2 Images:
            View_Results.Single_Viewer(Cloud_Offset,Cloud.number_of_cells,Estimated_Time,RMSE_Percent,RMSE_Meter,Mean_RMSE_Percent,Mean_RMSE_Meter)    
            
        #* Mean Result For All Images:
        View_Results.Mean_Viewer(Mean_Time,Mean_Points,Mean_RMSE_Percent,Mean_RMSE_Meter)


    def Plot_CSV(self):

        Plot=Plot_Viewer(self.Pars_Dict)
        Plot.Show_Plot()

    def Analyze_CSV(self):

        Plot=Plot_Viewer(self.Pars_Dict)
        Analyze=Plot.Analyze_Result()

        Show_Result=Result_Viewer(self.Pars_Dict)
        Show_Result.Mean_Viewer(round(Analyze[1],2),round(Analyze[0]),round(Analyze[3],2),round(Analyze[2],2))
