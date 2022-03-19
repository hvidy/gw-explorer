import dash
from dash import Dash, dcc, html, Input, Output, State
import dash_daq as daq
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from gwpy.timeseries import TimeSeries
from gwpy.time import tconvert
from gwpy.signal import filter_design
import pandas as pd
import numpy as np
from functools import lru_cache


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server


def transform_value(value):
    return 10 ** value

@lru_cache(maxsize=32)
def read_model(mass,q):
    mass = int(np.around(mass, decimals=0))
    q = np.around(q, decimals=1)
    fn = 'models/waveform-M{:d}q{:3.2f}.h5'.format(mass,q)
    return pd.read_hdf(fn)

@lru_cache(maxsize=32)
def get_data(fn,on1,on2,bprange0,bprange1):
    
    strain = read_data('data/'+fn)
    
    #Setup notch filters
    notches = [filter_design.notch(line, strain.sample_rate) for line in (60, 120, 180)]
    notch_zpk = filter_design.concatenate_zpks(*notches)

    if on1 & on2:
        bp = filter_design.bandpass(bprange0, bprange1, strain.sample_rate)
        zpk = filter_design.concatenate_zpks(bp, *notches)
        bpdata = strain.filter(zpk, filtfilt=True)
    elif on1:
        zpk = filter_design.concatenate_zpks(*notches)
        bpdata = strain.filter(zpk, filtfilt=True)
    elif on2:
        bp = filter_design.bandpass(bprange0, bprange1, strain.sample_rate)
        bpdata = strain.filter(bp, filtfilt=True)
    else:
        bpdata = strain
    
    bpdata = bpdata.crop(*bpdata.span.contract(1))
    sg = bpdata.spectrogram2(fftlength=1/16., overlap=15/256.) ** (1/2.)
    
    return bpdata,sg

@lru_cache(maxsize=32)
def read_data(fn):
    return TimeSeries.read(fn)


# Time series length
time_length = 5


#App Layout
app.layout = html.Div([
    
    html.H1("Gravitational Wave Data Explorer"),
    
    html.Div([
        
        html.H5('GW Event'),
        
        dcc.Dropdown(id='event-selector',
                     options=[
                         {'label':'GW150914','value':'GW150914-H1.hdf5'},
                         {'label':'GW170104','value':'GW170104-L1.hdf5'},
                         {'label':'GW170608','value':'GW170608-H1.hdf5'},
                         {'label':'GW170809','value':'GW170809-L1.hdf5'},
                         {'label':'GW170814','value':'GW170814-L1.hdf5'},
                         {'label':'GW190408_181802','value':'GW190408_181802-L1.hdf5'},
                         {'label':'GW190412','value':'GW190412-L1.hdf5'},
                         {'label':'GW190521_074359','value':'GW190521_074359-L1.hdf5'},
                         {'label':'GW190814','value':'GW190814-L1.hdf5'},
                         {'label':'GW200129_065458','value':'GW200129_065458-L1.hdf5'}
                         ],
                     value='GW150914-H1.hdf5'),
    
        dcc.Graph(id='timeseries')
        
    ], style={'display': 'inline-block', 
              'vertical-align':'top', 
#               'margin-left':'3vw', 
#               'margin-top':'3vw',
              'width':'60vw'
             }),
    
    html.Div([
    
        html.H3("Signal processing controls"),
        
        html.Div([
            html.Div([
                daq.BooleanSwitch(id='bandpass-filter-switch', 
                                  on=False, 
                                  label='Bandpass filter', 
                                  labelPosition="top",
                                  color="#5E5EFF"
                                 ),
            ],style={'display':'inline-block',
                     'vertical-align':'top'
                    }
            ),
            html.Div([
                daq.BooleanSwitch(id='notch-filter-switch', 
                                  on=False, 
                                  label='Notch filters', 
                                  labelPosition="top",
                                  color="#5E5EFF"
                                 )
            ],style={'display':'inline-block',
                     'vertical-align':'top',
                     'margin-left': '3vw'
                    }
            )
        ], className='row'),
            
        html.Div(id='output-container-range-slider-non-linear', style={'margin-top':20,'margin-bottom': 20}),
        
        dcc.RangeSlider(min=0, max=2.7,
                        id='bandpass-filter-range-slider',
                        marks={i: '{} Hz'.format(10 ** i) for i in range(3)},
                        value=[0, 2.7],
                        dots=False,
                        step=0.01,
                        allowCross=False,
                        updatemode='mouseup'
        ),
    
        html.Br(),
        
        html.H3("Model fitting controls"),
        
        html.Div([
            html.Div([
                daq.BooleanSwitch(id='model-switch', 
                                  on=False, 
                                  label='Plot model', 
                                  labelPosition="top",
                                  color="#FF5E5E"
                                 ),
            ],style={'display':'inline-block',
                     'vertical-align':'top',
                     'margin-top': '0.2vw'
                    }
            ),
            html.Div([
                html.P("Merger time"),
                dcc.Input(
                    id='model-time-offset',
                    type="number",
                    min=0,
                    max=time_length,
                    value=time_length/2
                )
            ],style={'display':'inline-block',
                     'vertical-align':'top',
                     'margin-left': '3vw'
                    }
            )
        ], className='row'),
        
        html.Div([
            html.Div([
                html.P("Chirp mass"),
                daq.Slider(id='chirp-mass-slider',
                           min=6,
                           value=20,
                           max=48,
                           step=2,
                           marks={i: u'{} M\u2609'.format(i) for i in range(10,50,10)},
                           color="#FF5E5E",
                           updatemode='mouseup'                      
                          )
            ],style={'vertical-align':'top',
                     'margin-left': '3vw',
                     'margin-top': '1vw',
                    }
            ),
            html.Div([
                html.P("Mass ratio"),
                daq.Slider(id='mass-ratio-slider',
                           min=0.1,
                           value=0.5,
                           max=1,
                           step=0.1,
                           marks={i: u'{:0.1f}'.format(i) for i in np.arange(0.,1.1,0.2)},
                           color="#FF5E5E",
                           updatemode='mouseup'                      
                          )
            ],style={'vertical-align':'top',
                     'margin-left': '3vw',
                     'margin-top': '2vw',
                    }
            ),
            html.Div([
                html.P("Amplitude scale"),
                daq.Slider(id='amplitude-slider',
                           min=-1,
                           value=-0.1,
                           max=1,
                           step=0.1,
                           marks={i: '{}'.format(10 ** i) for i in range(-1,2,1)},
                           color="#FF5E5E",
                           updatemode='mouseup'                      
                          )
            ],style={'vertical-align':'top',
                     'margin-left': '3vw',
                     'margin-top': '2vw',
                    }
            ),
            html.Div([
                html.P("Polarisation"),
                daq.Slider(id='phase-slider',
                           min=0,
                           value=0.5,
                           max=1,
                           step=0.01,
                           marks={"0":"+","1":"x"},
                           color="#FF5E5E",
                           updatemode='mouseup'
                          )
            ],style={'vertical-align':'top',
                     'margin-left': '3vw',
                     'margin-top': '2vw',
                    }
            ),
            
            html.Div([
                daq.LEDDisplay(id='chirp-mass-display',
                               color="#FF5E5E",
                               size=18,
                               label=u"Chirp mass (M\u2609)",
                               labelPosition="bottom"
                              )                
            ],style={'display':'inline-block',
                     'vertical-align':'top',
                     'width': '7vw',
                     'margin-top': '2vw'
                    }
            ),
            html.Div([
                daq.LEDDisplay(id='mass-ratio-display',
                               color="#FF5E5E",
                               size=18,
                               label='Mass ratio',
                               labelPosition="bottom"
                              )
            ],style={'display':'inline-block',
                     'vertical-align':'top',
                     'margin-left': '1vw',
                     'margin-top': '2vw',
                     'width': '7vw'
                    }
            ),
            html.Div([
                daq.LEDDisplay(id='amplitude-display',
                               color="#FF5E5E",
                               size=18,
                               label='Amplitude scale',
                               labelPosition="bottom"
                              )
            ],style={'display':'inline-block',
                     'vertical-align':'top',
                     'margin-left': '1vw',
                     'margin-top': '2vw',
                     'width': '7vw'
                    }
            ),

            
        ], className='row'),
    ], style={'display': 'inline-block', 
              'vertical-align':'top', 
              'margin-left':'3vw', 
#               'margin-top':'3vw',
              'width':'31vw'
             })
])

#Callbacks

@app.callback(
    Output('output-container-range-slider-non-linear', 'children'),
    Input('bandpass-filter-range-slider', 'value'))
def update_output(value):
    transformed_value = [transform_value(v) for v in value]
    return 'Filter passband: [{:0.2f}, {:0.2f}] Hz'.format(
        transformed_value[0],
        transformed_value[1]
    )

@app.callback(
    Output('chirp-mass-display','value'),
    Input('chirp-mass-slider','value'))
def update_output(value):
    return str(int(np.around(value, decimals=0)))

@app.callback(
    Output('mass-ratio-display','value'),
    Input('mass-ratio-slider','value'))
def update_output(value):
    return str(np.around(value, decimals=1))

@app.callback(
    Output('amplitude-display','value'),
    Input('amplitude-slider','value'))
def update_output(value):
    return str(np.around(transform_value(value), decimals=2))


@app.callback(
    Output('timeseries', 'figure'),
    Input('event-selector','value'),
    Input('notch-filter-switch','on'),
    Input('bandpass-filter-switch','on'),
    Input('bandpass-filter-range-slider','value'),
    Input('model-switch','on'),
    Input('chirp-mass-slider','value'),
    Input('mass-ratio-slider','value'),
    Input('model-time-offset','value'),
    Input('amplitude-slider','value'),
    Input('phase-slider','value'))
def update_figure(fn,on1,on2,value,model_on,mass,q,t0,A,phi):
    
    bprange = [transform_value(v) for v in value]
    Amp = transform_value(A)
    
    bpdata, sg = get_data(fn,on1,on2,bprange[0],bprange[1])
#     sg = calc_sg(bpdata)
        
    fig = make_subplots(rows=2, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.1
                       )
    
    fig.add_trace(go.Scatter(x=bpdata.times-bpdata.t0,
                  y=bpdata.value,showlegend=False,line={'color': '#5E5EFF'},hoverinfo='skip'),
                  row=1,col=1
                 )
    
    if model_on:
        #Load data for model
        df = read_model(mass,q)
        
        fig.add_trace(go.Scatter(x=df.time+float(t0),y=Amp*((1-phi)*df.Aorth-phi*df.Adiag)
                                 ,showlegend=False,line={'color': '#FF5E5E'},hoverinfo='skip'),
                      row=1,col=1)
        
        fig.add_trace(go.Scatter(x=df.time+float(t0),y=df.Freq,
                                 showlegend=False, line={'color': '#FF5E5E'},hoverinfo='skip'),
                      row=2,col=1)
    
    fig.add_trace(go.Heatmap(z=sg.value.T,
                             x=(sg.times-sg.t0).value,
                             y=sg.frequencies.value,
                             colorscale='Viridis',
                             colorbar=dict(title=u'Strain ASD [1/\u221aHz]',len=0.5, y=0.25), 
                             hoverinfo='skip'
                            ),row=2,col=1)
    
    fig.update_xaxes(rangeslider={'visible':True}, rangeslider_thickness=0.05, row=1, col=1)
    fig.update_xaxes(title_text='Time [seconds] from ' + tconvert(bpdata.t0.value).isoformat(' ') + ' UTC',
                     row=2,col=1)
    
    fig.update_yaxes(title_text='Strain Amplitude', row=1, col=1)
    fig.update_yaxes(title_text='Frequency (Hz)', type='log', row=2, col=1)

    fig.update_layout(height=800, uirevision='constant')
    
    return fig
    
# app.run_server(mode='external')

if __name__ == '__main__':
    app.run_server(debug=True)