from __future__ import division, print_function

import numpy as np
import pandas as pd

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
import os
init_notebook_mode()

np.set_printoptions(precision=2)

def calcGIW(datafile):

    if( ('cycGIW_XYZ' in datafile.columns) is True ):
            print('cycGIW_XYZ is already in the dataframe.')
            return

    def eihToGIW(rowIn):            

        # Grab gransformation matrix
        headTransform_4x4 = np.reshape(rowIn["viewMat_4x4"],[4,4])
        # Transpose
        headTransform_4x4 = headTransform_4x4.T

        # Grab cyc EIH direction
        cycEyeInHead_XYZ = rowIn['cycEyeInHead_XYZ']
        # Add a 1 to convert to homogeneous coordinates
        cycEyeInHead_XYZW = np.hstack( [cycEyeInHead_XYZ,1])

        # Take the dot product!
        cycGIWVec_XYZW = np.dot( headTransform_4x4,cycEyeInHead_XYZW)

        # Now, convert into a direction from the cyclopean eye in world coordinates
        # Also, we can discard the w term
        cycGIWDir_XYZ = (cycGIWVec_XYZW[0:3]-rowIn["viewPos_XYZ"]) / np.linalg.norm((cycGIWVec_XYZW[0:3]-rowIn["viewPos_XYZ"]))

        # You must return as a list or a tuple
        return list(cycGIWDir_XYZ)
    
    datafile['cycGIW_XYZ'] = datafile.apply(lambda row: eihToGIW(row),axis=1)
    
    return datafile
    


def calcAngularVelocity(datafile):

    if( ('cycGIW_XYZ' in datafile.columns) is False ):
        print('Missing GIW signal.  Calculating...')
        datafile = calcGIW(datafile)
        
    datafile['smiDateTime'] = pd.to_datetime(datafile.eyeTimeStamp,unit='ns')
    deltaTime = datafile['smiDateTime'].diff()
    deltaTime.loc[deltaTime.dt.microseconds==0] = pd.NaT
    deltaTime = deltaTime.fillna(method='bfill', limit=1)
    datafile['smiDeltaT'] = deltaTime.dt.microseconds / 1000000

    changeInGIW_fr = [ np.rad2deg(np.arccos(np.vdot(abc,xyz)))
                        for abc, xyz in zip(datafile['cycGIW_XYZ'], np.roll(datafile['cycGIW_XYZ'],1))]

    changeInEIH_fr = [ np.rad2deg(np.arccos(np.vdot(abc,xyz)))
                        for abc, xyz in zip(datafile['cycEyeInHead_XYZ'], np.roll(datafile['cycEyeInHead_XYZ'],1))]

    datafile['cycGIWVelocity'] = changeInGIW_fr / datafile['smiDeltaT']
    datafile['cycEIHVelocity'] = changeInEIH_fr / datafile['smiDeltaT']

    datafile['cycGIWVelocity'] = datafile['cycGIWVelocity'].fillna(method='bfill')
    datafile['cycEIHVelocity'] = datafile['cycEIHVelocity'].fillna(method='bfill')
    
    return datafile





def createHead(headTransform_4x4 = np.eye(4)):

        phi = np.linspace(0, 2*np.pi)
        theta = np.linspace(-np.pi/2, np.pi/2)
        phi, theta = np.meshgrid(phi, theta)

        x = np.cos(theta) * np.sin(phi) * .15
        y = np.sin(theta) * 0.2
        z = np.cos(theta) * np.cos(phi) * .15
        w = np.ones(2500)

        x = x.flatten()
        y = y.flatten()
        z = z.flatten()

        if headTransform_4x4 is False:
            headVertices_XYZW =np.array([x,y,z,w])
        else:
            headVertices_XYZW = np.dot( headTransform_4x4,[x,y,z,w])

        headShape = go.Mesh3d({ 'x':headVertices_XYZW[0,:], 
                      'y': headVertices_XYZW[2,:], 
                      'z': headVertices_XYZW[1,:],'alphahull': 0},
                      color='rgb(20, 145, 145)',
                      )

        return headShape


  

def plotGazeVelocity(datafile, trialNumber, columnNames,yLim=[0 ,500],width=800,height=600,inline=True):
    
    if( (trialNumber in datafile['trialNumber'].values) is False ):
        print('Trial number not found in data file.')
        return

    #trialNum = 13
    trialData = datafile.groupby(['trialNumber']).get_group((trialNumber))

    import plotly.plotly as py
    import plotly.graph_objs as go

    import pandas as pd

    traces = []
    
    colors_idx = ['rgb(0,204,204)','rgb(128,128,128)','rgb(204,0,0)','rgb(102,0,204)']

    time_fr = np.array(trialData['frameTime'] - trialData['frameTime'].iloc[0])

    for idx, columnName in enumerate(columnNames):
        
        scatterObj = go.Scatter(
        x=time_fr,
        y=trialData[columnName],
        name = columnName,
        line = dict(color = colors_idx[idx],width=3),
        opacity = 0.8)

        traces.append(scatterObj)
        
    ################################################################
    ## You can ignore this section.  This code adds event labels to the time series

    events_fr = trialData['eventFlag'].values
    eventIdx = np.where( trialData['eventFlag'] > 0 )

    eventTimes_idx =  time_fr[eventIdx]  

    eventText_idx = [str(event) for event in events_fr[np.where(events_fr>2)]]
    eventText_idx = np.array([[eT, '','',''] for eT in eventText_idx]).flatten()

    x = np.array([np.array([eT, eT, np.nan, np.nan]).flatten() for eT in eventTimes_idx]).flatten()
    y = np.tile([ (yLim[1]-yLim[0])*.95 ,yLim[0],np.nan,np.nan],len(eventTimes_idx))

    eventLabels = go.Scatter(
        x=x,
        y=y,
        mode='lines+text',
        name = "trial events",
        text=eventText_idx,
        textposition='top right',
        line = dict(color = ('rgba(30, 150, 30,.5)'),width = 4)
    )

    ################################################################

    layout = dict(
        dragmode= 'pan',
        title='Time Series with Rangeslider',
        width=width,
        height=height,
        yaxis=dict(range=yLim, title='velocity'),
        xaxis=dict(
            rangeslider=dict(),
            type='time',
            range=[0,1.0]
        )
    )
    
    traces.append(eventLabels)
    
    fig = dict(data=traces, layout=layout)
    return fig
    
def plotEIH(cycEyeInHead_XYZW,
            xRange = [-1,1],
            yRange = [-1,1],
            zRange = [-1,1],
            yLim=[0 ,500],
            width=800,
            height=600,
            inline=False):

    head = createHead()
    
    eihDir = go.Scatter3d(x=[0,cycEyeInHead_XYZW[0]],
                   y=[0,cycEyeInHead_XYZW[2]],
                   z=[0,cycEyeInHead_XYZW[1]],
                   mode='lines',
                   line = dict(
                       color = ('rgb(205, 12, 24)'),
                       width = 4)
                  )
    
    layout = go.Layout(title="Head Centered Space", 
                    width=width,
                    height=height,
                    showlegend=False,
                    scene=go.Scene(aspectmode='manual',
                                aspectratio=dict(x=1, y=1, z=1),
                                xaxis=dict(range=xRange, title='x Axis'),
                                yaxis=dict(range=yRange, title='y Axis'),
                                zaxis=dict(range=zRange, title='z Axis'),

                               ),
                    margin=go.Margin(t=100),
                    hovermode='closest',

                    )

    fig = go.Figure(data=go.Data([head,eihDir]),layout=layout)

    return fig
    
def plotGIW(viewPos_XYZ,
            cycGIW_XYZ,
            ballPos_XYZ,
            headTransform_4x4,
            xRange = [-1,1],
            yRange = [-1,1],
            zRange = [-1,1],
            width=800,
            height=600,
            inline=False):

    headShape = createHead(headTransform_4x4)
    
    giwDir = go.Scatter3d(x=[viewPos_XYZ[0],cycGIW_XYZ[0]],
                          y=[viewPos_XYZ[2],cycGIW_XYZ[2]],
                          z=[viewPos_XYZ[1],cycGIW_XYZ[1]],
                          mode='lines+text',
                    text=['','gaze'],
                    textposition='top right',
                    textfont=dict(
                        family='sans serif',
                        size=14,
                        color=('rgb(20, 0, 145)'),
                        ),
                    line = dict(
                       color=('rgb(20, 0, 145)'),
                       width = 4)
                         )

    xyz = np.subtract(ballPos_XYZ,viewPos_XYZ)
    ballDir_XYZ = xyz / np.linalg.norm(xyz)
    ballEndPoint_XYZ = viewPos_XYZ + ballDir_XYZ*1.5
    
    ballDir  = go.Scatter3d(x=[viewPos_XYZ[0],ballEndPoint_XYZ[0]],
        y=[viewPos_XYZ[2],ballEndPoint_XYZ[2]],
        z=[viewPos_XYZ[1],ballEndPoint_XYZ[1]],
        mode='lines+text',
        text=['','ball'],
        textposition='top right',
        textfont=dict(
            family='sans serif',
            size=14,
            color='rgb(30, 150, 30)',
            ),
        line = dict(
           color = ('rgb(30, 150, 30)'),
           width = 4)
                    )

    layout = go.Layout(title="Gaze in World", 
                    width=width,
                    height=height,
                    showlegend=False,
                    scene=go.Scene(aspectmode='manual',
                                aspectratio=dict(x=1, y=1, z=1),
                                xaxis=dict(range=xRange, title='x Axis'),
                                yaxis=dict(range=yRange, title='y Axis'),
                                zaxis=dict(range=zRange, title='z Axis'),

                               ),
                    margin=go.Margin(t=100),
                    hovermode='closest',

                    )

    fig=go.Figure(data=go.Data([giwDir,ballDir,headShape]),layout=layout)

    return fig


def gd_eihToGIW(rowIn):

    # Grab gransformation matrix
    headTransform_4x4 = np.reshape(rowIn["viewMat_4x4"],[4,4])
    # Transpose
    headTransform_4x4 = headTransform_4x4.T

    # Grab cyc EIH direction
    cycEyeInHead_XYZ = rowIn['cycEyeInHead_XYZ']
    # Add a 1 to convert to homogeneous coordinates
    cycEyeInHead_XYZW = np.hstack( [cycEyeInHead_XYZ,1])

    # Take the dot product!
    cycGIWVec_XYZW = np.dot( headTransform_4x4,cycEyeInHead_XYZW)
    
    # Now, convert into a direction from the cyclopean eye in world coordinates
    # Also, we can discard the w term
    cycGIWDir_XYZ = (cycGIWVec_XYZW[0:3]-rowIn["viewPos_XYZ"]) / np.linalg.norm((cycGIWVec_XYZW[0:3]-rowIn["viewPos_XYZ"]))
    
    # You must return as a list or a tuple
    return list(cycGIWDir_XYZ)