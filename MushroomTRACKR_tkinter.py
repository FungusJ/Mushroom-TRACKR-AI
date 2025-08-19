#!/usr/bin/env python3
"""
MushroomTRACKR - Tkinter Version
A fungal genus predictor with province-based temperature & humidity data
Converted from Streamlit to Tkinter while maintaining all functionality
"""
#TODO: dropdown list and fuzzy search

import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
import threading
from rapidfuzz import process, fuzz
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import pickle
import time

# === Province Coordinates ===
province_coordinates = {
    'Östergötland': (58.4064, 15.6255), 'Skåne': (55.9903, 13.5958), 'Uppsala': (59.8586, 17.6389),
    'Västra Götaland': (58.2528, 13.0596), 'Södermanland': (59.0336, 16.7519), 'Stockholm': (59.3293, 18.0686),
    'Jämtland': (63.1712, 14.9592), 'British Columbia': (53.7267, -127.6476), 'Ontario': (51.2538, -85.3232),
    'Quebec': (52.9399, -73.5491), 'Alberta': (53.9333, -116.5765), 'Manitoba': (49.8951, -97.1384),
    'Nova Scotia': (44.6820, -63.7443), 'Queensland': (-20.9176, 142.7028), 'New South Wales': (-33.8688, 151.2093),
    'Victoria': (-37.4713, 144.7852), 'Western Australia': (-27.6728, 121.6283), 'Tasmania': (-42.0351, 146.6367),
    'South Australia': (-30.0002, 136.2092), 'California': (36.7783, -119.4179), 'Texas': (31.9686, -99.9018),
    'Florida': (27.9944, -81.7603), 'New York': (43.0000, -75.0000), 'Oregon': (43.8041, -120.5542),
    'Washington': (47.7511, -120.7401), 'Arizona': (34.0489, -111.0937), 'Nevada': (38.8026, -116.4194),
    'Illinois': (40.6331, -89.3985), 'Massachusetts': (42.4072, -71.3824), 'Colorado': (39.5501, -105.7821),
    'England': (52.3555, -1.1743), 'Scotland': (56.4907, -4.2026), 'Wales': (52.1307, -3.7837),
    'Northern Ireland': (54.7877, -6.4923), 'Bavaria': (48.7904, 11.4979), 'North Rhine-Westphalia': (51.4332, 7.6616),
    'Hesse': (50.6521, 9.1624), 'Brandenburg': (52.4125, 12.5316), 'Berlin': (52.5200, 13.4050),
    'Yunnan': (25.0453, 102.7097), 'Guangdong': (23.3790, 113.7633), 'Sichuan': (30.5728, 104.0668),
    'Beijing': (39.9042, 116.4074), 'Shanghai': (31.2304, 121.4737), 'Tibet': (31.6927, 88.0924),
    'São Paulo': (-23.5505, -46.6333), 'Amazonas': (-3.4653, -62.2159), 'Paraná': (-24.8949, -51.5500),
    'Bahia': (-12.5797, -41.7007), 'Rio de Janeiro': (-22.9068, -43.1729), 'Minas Gerais': (-18.5122, -44.5550),
    'Western Cape': (-33.2278, 21.8569), 'Eastern Cape': (-32.2968, 26.4194), 'KwaZulu-Natal': (-28.5306, 30.8958),
    'Gauteng': (-26.2708, 28.1123), 'Chiapas': (16.7569, -93.1292), 'Jalisco': (20.6597, -103.3496),
    'Yucatán': (20.7099, -89.0943), 'Oaxaca': (17.0732, -96.7266), 'Kerala': (10.8505, 76.2711),
    'Tamil Nadu': (11.1271, 78.6569), 'Maharashtra': (19.7515, 75.7139), 'West Bengal': (22.9868, 87.8550),
    'Assam': (26.2006, 92.9376), 'Moscow': (55.7558, 37.6173), 'Siberia': (61.0137, 99.1967),
    'Primorsky Krai': (45.0525, 135.0000), 'Java': (-7.4910, 110.0044), 'Bali': (-8.3405, 115.0920),
    'Sumatra': (-0.5897, 101.3431), 'Luzon': (16.5000, 121.0000), 'Mindanao': (8.0000, 125.0000),
    'Visayas': (11.5000, 123.5000), 'Buenos Aires': (-34.6037, -58.3816), 'Santa Fe': (-31.5855, -60.7238),
    'Córdoba': (-31.4201, -64.1888)
}

class MushroomTrackerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🍄 MushroomTRACKR - Fungal Genus Predictor")
        self.root.geometry("1000x700")
        
        # Data storage
        self.biodata = None
        self.tempdata = None
        self.final_model = None
        self.scaler = None
        self.le_genus = None
        self.features = None
        self.global_mean_temp = None
        self.biodata_top = None
        self.acc = None
        self.bal_acc = None
        
        # Label encoders
        self.le_stateProvince = None
        self.le_family = None
        self.le_country = None
        
        # Create GUI
        self.create_widgets()
        
        # Load data on startup
        self.load_data_async()
    
    def create_widgets(self):
        """Create the main GUI components"""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create tabs
        self.create_home_tab()
        self.create_data_tab()
        self.create_predict_tab()
        self.create_about_tab()
    
    def create_home_tab(self):
        """Create the home/welcome tab"""
        self.home_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.home_frame, text="Home")
        
        # Title
        title_label = ttk.Label(self.home_frame, text="🍄 MushroomTRACKR", 
                               font=("Arial", 24, "bold"))
        title_label.pack(pady=20)
        
        subtitle_label = ttk.Label(self.home_frame, text="Fungal Genus Predictor with Province-based Data", 
                                  font=("Arial", 14))
        subtitle_label.pack(pady=10)
        
        # Status frame
        self.status_frame = ttk.LabelFrame(self.home_frame, text="System Status", padding=10)
        self.status_frame.pack(fill='x', padx=20, pady=10)
        
        self.status_label = ttk.Label(self.status_frame, text="Loading data...", 
                                     font=("Arial", 10))
        self.status_label.pack()
        
        self.progress_bar = ttk.Progressbar(self.status_frame, mode='indeterminate')
        self.progress_bar.pack(fill='x', pady=5)
        
        # Quick actions
        actions_frame = ttk.LabelFrame(self.home_frame, text="Quick Actions", padding=10)
        actions_frame.pack(fill='x', padx=20, pady=10)
        
        ttk.Button(actions_frame, text="View Data", 
                  command=lambda: self.notebook.select(1)).pack(side='left', padx=5)
        ttk.Button(actions_frame, text="Make Predictions", 
                  command=lambda: self.notebook.select(2)).pack(side='left', padx=5)
    
    def create_data_tab(self):
        """Create the data viewing tab"""
        self.data_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.data_frame, text="Data")
        
        # Controls frame
        controls_frame = ttk.Frame(self.data_frame)
        controls_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(controls_frame, text="Refresh Data", 
                  command=self.refresh_data).pack(side='left', padx=5)
        ttk.Button(controls_frame, text="Export Data", 
                  command=self.export_data).pack(side='left', padx=5)
        
        # Treeview for data display
        self.data_tree = ttk.Treeview(self.data_frame)
        self.data_tree.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Scrollbars
        vsb = ttk.Scrollbar(self.data_frame, orient="vertical", command=self.data_tree.yview)
        vsb.pack(side='right', fill='y')
        self.data_tree.configure(yscrollcommand=vsb.set)
        
        hsb = ttk.Scrollbar(self.data_frame, orient="horizontal", command=self.data_tree.xview)
        hsb.pack(side='bottom', fill='x')
        self.data_tree.configure(xscrollcommand=hsb.set)
    
    def create_predict_tab(self):
        """Create the prediction tab"""
        self.predict_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.predict_frame, text="Predict")
        
        # Input frame
        input_frame = ttk.LabelFrame(self.predict_frame, text="Input Parameters", padding=10)
        input_frame.pack(fill='x', padx=10, pady=5)
        
        # Country input
        ttk.Label(input_frame, text="Country:").grid(row=0, column=0, sticky='w', padx=5, pady=5)
        self.country_var = tk.StringVar()
        self.country_entry = ttk.Entry(input_frame, textvariable=self.country_var, width=30)
        self.country_entry.grid(row=0, column=1, padx=5, pady=5)
        
        # Province input
        ttk.Label(input_frame, text="Province/State:").grid(row=1, column=0, sticky='w', padx=5, pady=5)
        self.province_var = tk.StringVar()
        self.province_entry = ttk.Entry(input_frame, textvariable=self.province_var, width=30)
        self.province_entry.grid(row=1, column=1, padx=5, pady=5)
        
        # Family selection
        ttk.Label(input_frame, text="Family (optional):").grid(row=2, column=0, sticky='w', padx=5, pady=5)
        self.family_var = tk.StringVar()
        self.family_entry = ttk.Entry(input_frame, textvariable=self.family_var, width=30)
        self.family_entry.grid(row=2, column=1, padx=5, pady=5)
        
        # Top K selection
        ttk.Label(input_frame, text="Top K Results:").grid(row=3, column=0, sticky='w', padx=5, pady=5)
        self.top_k_var = tk.StringVar(value="10")
        self.top_k_spin = ttk.Spinbox(input_frame, from_=1, to=50, textvariable=self.top_k_var, width=5)
        self.top_k_spin.grid(row=3, column=1, sticky='w', padx=5, pady=5)
        
        # Predict button
        ttk.Button(input_frame, text="Predict Genera", 
                  command=self.make_prediction).grid(row=4, column=0, columnspan=2, pady=10)
        
        # Results frame
        results_frame = ttk.LabelFrame(self.predict_frame, text="Prediction Results", padding=10)
        results_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Results treeview
        self.results_tree = ttk.Treeview(results_frame, columns=('Rank', 'Genus', 'Probability'), 
                                        show='headings', height=15)
        self.results_tree.heading('Rank', text='Rank')
        self.results_tree.heading('Genus', text='Genus')
        self.results_tree.heading('Probability', text='Probability')
        self.results_tree.column('Rank', width=50)
        self.results_tree.column('Genus', width=200)
        self.results_tree.column('Probability', width=100)
        self.results_tree.pack(fill='both', expand=True)
        
        # Scrollbars for results
        vsb = ttk.Scrollbar(results_frame, orient="vertical", command=self.results_tree.yview)
        vsb.pack(side='right', fill='y')
        self.results_tree.configure(yscrollcommand=vsb.set)
    
    def create_about_tab(self):
        """Create the about tab"""
        self.about_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.about_frame, text="About")
        
        about_text = """
MushroomTRACKR - Fungal Genus Predictor

This application predicts fungal genera based on:
- Province/State location
- Temperature and humidity data
- Family classification
- Machine learning models

Features:
- Random Forest classifier with 95%+ accuracy
- Province-based environmental data
- Interactive predictions
- Data visualization

Version: 1.0 (Tkinter Edition)
        """
        
        text_widget = tk.Text(self.about_frame, wrap='word', height=20, width=80)
        text_widget.pack(padx=10, pady=10, fill='both', expand=True)
        text_widget.insert('1.0', about_text)
        text_widget.config(state='disabled')
    
    def load_data_async(self):
        """Load data in background thread"""
        self.progress_bar.start()
        threading.Thread(target=self.load_data, daemon=True).start()
    
    def load_data(self):
        """Load and process all data"""
        try:
            # Update status
            self.root.after(0, lambda: self.status_label.config(text="Loading biodiversity data..."))
            
            # Load data (similar to original code but adapted)
            script_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Load biodiversity data
            filename1 = os.path.join(script_dir, 'biodiversity_data')
            filename2 = os.path.join(script_dir, '0010088-250415084134356.csv')
            temp_filename = os.path.join(script_dir, 'temperature_2m.csv')
            desired_cols = ['genus', 'stateProvince', 'family', 'phylum', 'species']
            
            # Load and process data
            biodata1 = self.load_biodata(filename1, desired_cols)
            biodata2 = self.load_biodata(filename2, desired_cols)
            self.biodata = pd.concat([biodata1, biodata2], ignore_index=True)
            
            self.tempdata = self.load_tempdata(temp_filename)
            
            # Clean and label data
            self.biodata, self.le_stateProvince, self.le_family, self.le_country = self.clean_and_label(
                self.biodata, province_coordinates)
            
            self.biodata = self.add_temp_humidity(self.biodata, self.tempdata, province_coordinates)
            
            # Train model
            self.final_model, self.scaler, self.le_genus, self.features, self.global_mean_temp, \
            self.biodata_top, self.acc, self.bal_acc = self.train_model(self.biodata)
            
            # Update UI
            self.root.after(0, self.update_status_success)
            
        except Exception as error:
            self.root.after(0, lambda: self.update_status_error(str(error)))
    
    def update_status_success(self):
        """Update status when loading is complete"""
        self.progress_bar.stop()
        self.status_label.config(
            text=f"Data loaded successfully!\n"
                 f"Model Accuracy: {self.acc:.2%}\n"
                 f"Balanced Accuracy: {self.bal_acc:.2%}")
        
        # Enable prediction tab
        self.refresh_data()
    
    def update_status_error(self, error):
        """Update status on error"""
        self.progress_bar.stop()
        self.status_label.config(text=f"Error loading data: {error}")
        messagebox.showerror("Error", f"Failed to load data: {error}")
    
    def load_biodata(self, filename, desired_cols):
        """Load biodiversity data"""
        try:
            biodata = pd.read_csv(filename, sep='\t', usecols=desired_cols, low_memory=False)
            biodata = biodata[biodata['phylum'] != 'Ascomycota']
            biodata.dropna(subset=['stateProvince', 'family', 'species'], inplace=True)
            return biodata
        except:
            # Return empty DataFrame if file not found
            return pd.DataFrame(columns=desired_cols)
    
    def load_tempdata(self, filename):
        """Load temperature data"""
        try:
            tempdata = pd.read_csv(filename)
            if 'humidity' not in tempdata.columns:
                tempdata['humidity'] = np.nan
            return tempdata
        except:
            # Return empty DataFrame if file not found
            return pd.DataFrame(columns=['latitude', 'longitude', 'temperature_C', 'humidity'])
    
    def clean_and_label(self, biodata, province_coordinates):
        """Clean and label the data"""
        from rapidfuzz import process, fuzz
        
        def fuzzy_group(text_series, threshold=95):
            mapping = {}
            unique_texts = text_series.unique()
            for i, text in enumerate(unique_texts):
                if text in mapping:
                    continue
                mapping[text] = text
                for other in unique_texts[i+1:]:
                    if fuzz.ratio(text.lower(), other.lower()) > threshold:
                        mapping[other] = text
            return text_series.map(mapping)
        
        # Clean data
        biodata['stateProvince_cleaned'] = fuzzy_group(biodata['stateProvince'], threshold=95)
        biodata['family_cleaned'] = fuzzy_group(biodata['family'], threshold=85)
        
        # Map provinces to countries (complete mapping from temp.py)
        province_to_country = {
            # Sweden
            'Östergötland': 'Sweden', 'Skåne': 'Sweden', 'Uppsala': 'Sweden', 'Västra Götaland': 'Sweden', 
            'Södermanland': 'Sweden', 'Stockholm': 'Sweden', 'Jämtland': 'Sweden', 'Göteborg': 'Sweden',
            'Torne lappmark': 'Sweden', 'Uppland': 'Sweden', 'Halland': 'Sweden', 'Småland': 'Sweden',
            'Västergötland': 'Sweden', 'Dalsland': 'Sweden', 'Öland': 'Sweden', 'Norrbotten': 'Sweden',
            'Medelpad': 'Sweden', 'Närke': 'Sweden', 'Lule lappmark': 'Sweden', 'Västmanland': 'Sweden',
            'Åsele lappmark': 'Sweden', 'Värmland': 'Sweden', 'Dalarna': 'Sweden', 'Gotland': 'Sweden',
            'Gästrikland': 'Sweden', 'Ångermanland': 'Sweden', 'Blekinge': 'Sweden', 'Bohuslän': 'Sweden',
            'Hälsingland': 'Sweden', 'Härjedalen': 'Sweden', 'Lappland': 'Sweden', 'Pite lappmark': 'Sweden',
            'Västerbotten': 'Sweden',

            # Finland
            'Pohjois-Savo': 'Finland', 'Etelä-Häme': 'Finland', 'Pohjois-Karjala': 'Finland',
            'Pohjois-Häme': 'Finland', 'Perä-Pohjanmaa': 'Finland', 'Enontekiön Lappi': 'Finland',
            'Varsinais-Suomi': 'Finland', 'Uusimaa': 'Finland', 'Kittilän Lappi': 'Finland',
            'Satakunta': 'Finland', 'Kainuu': 'Finland', 'Sompion Lappi': 'Finland', 'Oulun Pohjanmaa': 'Finland',
            'Koillismaa': 'Finland', 'Keski-Pohjanmaa': 'Finland', 'Etelä-Karjala': 'Finland',
            'Etelä-Savo': 'Finland', 'Etelä-Pohjanmaa': 'Finland', 'Åland (Ahvenanmaa)': 'Finland',
            'Inarin Lappi': 'Finland', 'Tartumaa': 'Finland', 'Läänemaa': 'Finland',

            # Norway
            'Østfold': 'Norway', 'Hedmark': 'Norway', 'Oppland': 'Norway', 'Nordland': 'Norway',
            'Sogn og Fjordane': 'Norway', 'Finnmark': 'Norway', 'Vestfold': 'Norway', 'Troms': 'Norway',
            'Buskerud': 'Norway', 'Telemark': 'Norway', 'Akershus/Oslo': 'Norway', 'Sør-Trøndelag': 'Norway',
            'Nord-Trøndelag': 'Norway', 'Møre og Romsdal': 'Norway', 'Lycksele lappmark': 'Norway',
            'Svalbard': 'Norway', 'Bjørnøya': 'Norway', 'Aust-Agder': 'Norway', 'Vest-Agder': 'Norway',
            'Rogaland': 'Norway', 'Hordaland': 'Norway',

            # Denmark
            'Fyn/Langeland': 'Denmark', 'Sydjylland': 'Denmark', 'Sjælland': 'Denmark',
            'Nordjylland': 'Denmark', 'Bornholm': 'Denmark', 'Jylland': 'Denmark', 'Lolland/Falster/Møn': 'Denmark',
            'Østjylland': 'Denmark', 'Vestjylland': 'Denmark',

            # Iceland
            'Austur-Island': 'Iceland', 'Nordur-Island': 'Iceland', 'Sudur-Island': 'Iceland', 
            'Vestur-Island': 'Iceland', 'Nordvestur-Island': 'Iceland',

            # Estonia
            'Pärnumaa': 'Estonia', 'Harjumaa': 'Estonia', 'Valgamaa': 'Estonia', 'Saaremaa': 'Estonia',
            'Järvamaa': 'Estonia', 'Viljandimaa': 'Estonia', 'Lääne-Virumaa': 'Estonia', 'Ida-Virumaa': 'Estonia',
            'Jõgevamaa': 'Estonia', 'Põlvamaa': 'Estonia', 'Võrumaa': 'Estonia', 'Raplamaa': 'Estonia',

            # Latvia
            'Riga': 'Latvia',

            # Lithuania
            'Vilnius': 'Lithuania',

            # Germany
            'Bayern': 'Germany', 'Sachsen-Anhalt': 'Germany', 'Rheinland-Pfalz': 'Germany',
            'Nordrhein-Westfalen': 'Germany', 'Baden-Württemberg': 'Germany', 'Schleswig-Holstein': 'Germany',
            'Hamburg': 'Germany', 'Thüringen': 'Germany', 'Niedersachsen': 'Germany', 'Saarland': 'Germany',
            'Mecklenburg-Vorpommern': 'Germany', 'Bremen': 'Germany', 'Berlin': 'Germany', 'Brandenburg': 'Germany',
            'Sachsen': 'Germany', 'Hesse': 'Germany', 'Hesse': 'Germany', 'North Rhine-Westphalia': 'Germany',
            'Bavaria': 'Germany',

            # France
            'Guadeloupe': 'France', 'île-de-France': 'France', 'Rhône-Alpes': 'France', 'Franche-Comté': 'France',
            'Picardie': 'France', 'Auvergne': 'France', 'Réunion': 'France', 'Sud-Ouest': 'France',
            'Midi-Pyrénées': 'France', 'Corse': 'France', 'Alsace': 'France', 'Pays-de-la-Loire': 'France',
            'Lorraine': 'France', 'Haute-Normandie': 'France', 'Bourgogne': 'France', 'Limousin': 'France',
            'Aquitaine': 'France', 'Champagne-Ardenne': 'France', 'Basse-Normandie': 'France',
            'Nord-Pas-de-Calais': 'France', 'Martinique': 'France', 'French Polynesia': 'France',
            'Saint Barthélémy': 'France', 'Bretagne': 'France', 'Provence-Alpes-Côte d’Azur': 'France',
            'Centre': 'France', 'Sud': 'France', 'Languedoc-Roussillon': 'France', 'Jura': 'France',
            'Corsica': 'France', 'Île-de-France': 'France', 'Normandie': 'France', 'Poitou-Charentes': 'France',

            # Switzerland
            'Graubünden': 'Switzerland', 'Eastern Switzerland': 'Switzerland', 'Bern': 'Switzerland', 
            'Neuchâtel': 'Switzerland', 'Schwyz': 'Switzerland', 'Zürich': 'Switzerland', 'Thurgau': 'Switzerland',
            'Luzern': 'Switzerland', 'Solothurn': 'Switzerland', 'Appenzell Ausserrhoden': 'Switzerland',
            'Aargau': 'Switzerland', 'Basel-Stadt': 'Switzerland', 'Basel-Landschaft': 'Switzerland',
            'Obwalden': 'Switzerland', 'Nidwalden': 'Switzerland', 'Uri': 'Switzerland', 'Zug': 'Switzerland',
            'Ticino': 'Switzerland', 'Vaud': 'Switzerland', 'Valais': 'Switzerland', 'Sankt Gallen': 'Switzerland',
            'Schaffhausen': 'Switzerland', 'Genève': 'Switzerland', 'Glarus': 'Switzerland', 'Fribourg': 'Switzerland',

            # Austria
            'Salzburg': 'Austria', 'Niederösterreich': 'Austria', 'Vorarlberg': 'Austria', 'Burgenland': 'Austria',
            'Oberösterreich': 'Austria', 'Tirol': 'Austria', 'Wien': 'Austria', 'Steiermark': 'Austria',
            'Kärnten': 'Austria',

            # UK
            'England': 'United Kingdom', 'Scotland': 'United Kingdom', 'Wales': 'United Kingdom',
            'Northern Ireland': 'United Kingdom', 'Channel Islands': 'United Kingdom',

            # Canada
            'Ontario': 'Canada', 'Quebec': 'Canada', 'British Columbia': 'Canada', 'Alberta': 'Canada',
            'Manitoba': 'Canada', 'Nova Scotia': 'Canada', 'Saskatchewan': 'Canada', 'New Brunswick': 'Canada',
            'Prince Edward Island': 'Canada', 'Newfoundland and Labrador': 'Canada', 'Yukon Territory': 'Canada',
            'Nunavut': 'Canada', 'Québec': 'Canada', 'Northwest Territories': 'Canada', 'Mid-Island': 'Canada',

            # USA
            'Wyoming': 'USA', 'Florida': 'USA', 'California': 'USA', 'Texas': 'USA', 'Oregon': 'USA',
            'Washington': 'USA', 'Illinois': 'USA', 'Massachusetts': 'USA', 'Colorado': 'USA', 'New York': 'USA',
            'Arizona': 'USA', 'Nevada': 'USA', 'Minnesota': 'USA', 'Indiana': 'USA', 'New Mexico': 'USA',
            'District of Columbia': 'USA', 'Alaska': 'USA', 'Hawaii': 'USA', 'Rhode Island': 'USA',
            'Connecticut': 'USA', 'Delaware': 'USA', 'Oklahoma': 'USA', 'Georgia': 'USA', 'Virginia': 'USA',
            'South Dakota': 'USA', 'West Virginia': 'USA', 'Wisconsin': 'USA', 'Arkansas': 'USA', 'Kentucky': 'USA',
            'Molokai': 'USA', 'Saint Croix': 'USA', 'Guam - Marianas': 'USA', 'Maine': 'USA', 'North Dakota': 'USA',
            'Nebraska': 'USA', 'Michigan': 'USA', 'Louisiana': 'USA', 'Maryland': 'USA', 'Montana': 'USA',
            'Utah': 'USA', 'Mississippi': 'USA', 'Iowa': 'USA', 'South Carolina': 'USA', 'Oahu': 'USA',
            'Lanai': 'USA', 'Kansas': 'USA', 'New Hampshire': 'USA', 'New Jersey': 'USA', 'Virgin Islands': 'USA',
            'Alabama': 'USA', 'Tennessee': 'USA', 'Missouri': 'USA', 'Pennsylvania': 'USA', 'North Carolina': 'USA',
            'Vermont': 'USA', 'Idaho': 'USA', 'Ohio': 'USA', 'New Mexico': 'USA', 'South Carolina': 'USA',
            'North Dakota': 'USA', 'South Dakota': 'USA', 'West Virginia': 'USA', 'Wisconsin': 'USA',
            'Arkansas': 'USA', 'Kentucky': 'USA', 'Mississippi': 'USA', 'Iowa': 'USA', 'Connecticut': 'USA',
            'Rhode Island': 'USA', 'Delaware': 'USA', 'Oklahoma': 'USA', 'Georgia': 'USA', 'Virginia': 'USA',
            'Alaska': 'USA', 'Hawaii': 'USA', 'Guam - Marianas': 'USA', 'Saint Croix': 'USA', 'Molokai': 'USA',
            'Lanai': 'USA', 'Oahu': 'USA', 'Virgin Islands': 'USA', 'New Jersey': 'USA', 'New Hampshire': 'USA',
            'Kansas': 'USA', 'Maine': 'USA', 'Nebraska': 'USA', 'Michigan': 'USA', 'Louisiana': 'USA',
            'Maryland': 'USA', 'Montana': 'USA', 'Utah': 'USA', 'Arizona': 'USA', 'Nevada': 'USA',
            'Colorado': 'USA', 'Illinois': 'USA', 'Massachusetts': 'USA', 'Oregon': 'USA', 'Washington': 'USA',
            'Texas': 'USA', 'Florida': 'USA', 'California': 'USA', 'New York': 'USA', 'Minnesota': 'USA',
            'Indiana': 'USA',

            # Brazil
            'Rio de Janeiro': 'Brazil', 'São Tomé': 'Brazil', 'Santa Catarina': 'Brazil', 'Amazonas': 'Brazil',
            'Goiás': 'Brazil', 'Pará': 'Brazil', 'Tocantins': 'Brazil', 'Amapá': 'Brazil', 'Roraima': 'Brazil',
            'Rondônia': 'Brazil', 'Alagoas': 'Brazil', 'Pernambuco': 'Brazil', 'Sergipe': 'Brazil',
            'Espírito Santo': 'Brazil', 'Mato Grosso do Sul': 'Brazil', 'Brasília Distrito Federal': 'Brazil',
            'Rio Grande do Sul': 'Brazil', 'Mato Grosso': 'Brazil', 'São Paulo': 'Brazil', 'Bahia': 'Brazil',
            'Piauí': 'Brazil', 'Acre': 'Brazil', 'Ceará': 'Brazil', 'Rio Grande do Norte': 'Brazil',
            'Paraná': 'Brazil', 'Minas Gerais': 'Brazil',

            # Mexico
            'Distrito Federal': 'Mexico', 'México (incl. Distrito Federal)': 'Mexico', 'Jalisco': 'Mexico',
            'Sonora': 'Mexico', 'Hidalgo': 'Mexico', 'Morelos': 'Mexico', 'Chihuahua': 'Mexico',
            'Durango': 'Mexico', 'Nuevo Leon': 'Mexico', 'Coahuila de Zaragoza': 'Mexico', 'Puebla': 'Mexico',
            'San Luis Potosi': 'Mexico', 'Tamaulipas': 'Mexico', 'Veracruz-Llave': 'Mexico', 'Yucatan': 'Mexico',
            'Zacatecas': 'Mexico', 'Colima': 'Mexico', 'Aguascalientes': 'Mexico', 'Guanajuato': 'Mexico',
            'Michoacan de Ocampo': 'Mexico', 'Nayarit': 'Mexico', 'Oaxaca': 'Mexico', 'Quintana Roo': 'Mexico',
            'Sinaloa': 'Mexico', 'Baja California': 'Mexico', 'Baja California Sur': 'Mexico', 'Campeche': 'Mexico',
            'Tabasco': 'Mexico', 'Tlaxcala': 'Mexico', 'Querétaro': 'Mexico', 'Chiapas': 'Mexico',
            'Yucatán': 'Mexico', 'Jalisco': 'Mexico', 'Oaxaca': 'Mexico',

            # Argentina
            'Jujuy': 'Argentina', 'Buenos Aires': 'Argentina', 'Entre Ríos': 'Argentina', 'Corrientes': 'Argentina',
            'Tucumán': 'Argentina', 'Salta': 'Argentina', 'Santa Fe': 'Argentina', 'Chubut': 'Argentina',
            'Neuquén': 'Argentina', 'Formosa': 'Argentina', 'Catamarca': 'Argentina', 'La Rioja': 'Argentina',
            'Misiones': 'Argentina', 'Santiago del Estero': 'Argentina', 'San Luis': 'Argentina',
            'Córdoba': 'Argentina', 'Rio Negro': 'Argentina', 'La Pampa': 'Argentina',

            # Chile
            'Isla Robinson Crusoe (Más a Tierra)': 'Chile', 'Valparaíso': 'Chile', 'Magallanes (excl. Antarctica Chilena)': 'Chile',
            'Bío-Bío': 'Chile', 'Araucania': 'Chile', 'Coquimbo': 'Chile', 'Los Lagos': 'Chile', 'Maule': 'Chile',
            'O’Higgins': 'Chile', 'Santiago, Region Metropolitana': 'Chile', 'Tarapacá': 'Chile', 'Antofagasta': 'Chile',
            'Atacama': 'Chile', 'Aisén del General Carlos Ibanez del Campo': 'Chile', 'Arica y Parinacota': 'Chile',
            'Easter Island': 'Chile', 'Isla Alejandro Selkirk (Más Afuera)': 'Chile', 'Los Rios': 'Chile',
            'Concepción': 'Chile',

            # Spain
            'Galicia': 'Spain', 'Castilla y León': 'Spain', 'Cataluña': 'Spain', 'Madrid': 'Spain',
            'Andalucia': 'Spain', 'Asturias': 'Spain', 'Cantabria': 'Spain', 'Navarra': 'Spain', 'Aragón': 'Spain',
            'Castilla-La Mancha': 'Spain', 'Comunidad Valenciana': 'Spain', 'Extremadura': 'Spain', 'Murcia': 'Spain',
            'La Rioja': 'Spain', 'Pais Vasco': 'Spain', 'Baleares': 'Spain', 'Tenerife': 'Spain',
            'Gran Canaria': 'Spain', 'La Palma': 'Spain', 'Isla de la Juventud': 'Cuba', 'Isla Alejandro Selkirk (Más Afuera)': 'Chile',

            # Poland
            'Opolskie': 'Poland', 'Lubuskie': 'Poland', 'Dolnoslaskie': 'Poland', 'Malopolskie': 'Poland',
            'Mazowieckie': 'Poland', 'Podlaskie': 'Poland', 'Pomorskie': 'Poland', 'Slaskie': 'Poland',
            'Swietokrzyskie': 'Poland', 'Warminsko-Mazurskie': 'Poland', 'Wielkopolskie': 'Poland',
            'Zachodniopomorskie': 'Poland', 'Kujawsko-Pomorskie': 'Poland', 'Lubelskie': 'Poland',
            'Lodzkie': 'Poland', 'Podkarpackie': 'Poland',

            # Czech Republic
            'Ustecky kraj': 'Czech Republic', 'Olomoucky kraj': 'Czech Republic', 'Jihocesky kraj': 'Czech Republic',
            'Jihomoravsky kraj': 'Czech Republic', 'Stredocesky kraj': 'Czech Republic', 'Karlovarsky kraj': 'Czech Republic',
            'Pardubicky kraj': 'Czech Republic', 'Plzensky kraj': 'Czech Republic', 'Moravskoslezsky kraj': 'Czech Republic',
            'Liberecky kraj': 'Czech Republic', 'Zlinsky kraj': 'Czech Republic', 'Královéhradecky kraj': 'Czech Republic',
            'Vysocina': 'Czech Republic', 'Praha': 'Czech Republic',

            # Hungary
            'Pest (incl. Budapest)': 'Hungary', 'Bács-Kiskun': 'Hungary', 'Tolna': 'Hungary', 'Györ-Moson-Sopron': 'Hungary',
            'Jász-Nagykun-Szolnok': 'Hungary', 'Szabolcs-Szatmár-Bereg': 'Hungary', 'Baranya': 'Hungary',
            'Fejér': 'Hungary', 'Veszprém': 'Hungary', 'Vas': 'Hungary', 'Komárom-Esztergom': 'Hungary',
            'Heves': 'Hungary', 'Nógrád': 'Hungary', 'Békés': 'Hungary', 'Csongrád': 'Hungary',
            'Somogy': 'Hungary', 'Zala': 'Hungary', 'Borsod-Abaúj-Zemplén': 'Hungary', 'Hajdú-Bihar': 'Hungary',

            # Slovakia
            'Banskobystricky kraj': 'Slovakia', 'Presovsky kraj': 'Slovakia', 'Bratislavsky kraj': 'Slovakia',
            'Trenciansky kraj': 'Slovakia', 'Trnavsky kraj': 'Slovakia', 'Nitriansky kraj': 'Slovakia',
            'Kosicky kraj': 'Slovakia',

            # Belarus
            'Magilev': 'Belarus', 'Brest': 'Belarus', 'Grodna': 'Belarus', 'Minsk': 'Belarus',

            # Armenia
            'Yerevan': 'Armenia', 'Kotayk': 'Armenia',

            # Georgia
            'Tbilisi': 'Georgia', 'Ajaria': 'Georgia', 'Kakheti': 'Georgia', 'Imereti': 'Georgia',
            'Samtskhe-Javakheti': 'Georgia', 'Kvemo Kartli': 'Georgia', 'Samegrelo-Zemo Svaneti': 'Georgia',
            'Mtskheta-Mtianeti': 'Georgia', 'Shida Kartli': 'Georgia',

            # Cyprus
            'Lemesos - Limassol': 'Cyprus', 'Nicosia - Lefkosia - Lefkosa': 'Cyprus', 'Paphos - Pafos - Baf': 'Cyprus',
            'Larnaca - Larnaka': 'Cyprus', 'Famagusta - Gazimagusa - Ammochostos': 'Cyprus',
            'Kyrenia - Girne - Keryneia': 'Cyprus',

            # Israel
            'Tel Aviv': 'Israel', 'Jerusalem': 'Israel', 'Haifa': 'Israel', 'Northern': 'Israel',
            'Central': 'Israel', 'Southern': 'Israel',

            # Morocco
            'Tunis': 'Morocco', 'Setif': 'Morocco', 'Chefchaouen': 'Morocco', 'Azilal': 'Morocco',
            'Al Haouz': 'Morocco', 'Agadir Ida-Outanane': 'Morocco', 'Marrakech': 'Morocco',
            'Meknès': 'Morocco', 'Rabat': 'Morocco', 'Boulmane': 'Morocco',

            # Tunisia
            'Tunis': 'Tunisia', 'Sousse': 'Tunisia', 'Siliana': 'Tunisia', 'Kasserine': 'Tunisia',
            'Ben Arous': 'Tunisia', 'Jendouba': 'Tunisia', 'Nabeul': 'Tunisia', 'Kef': 'Tunisia',

            # Egypt
            'Al Qalyubiyah': 'Egypt', 'Al Qahirah': 'Egypt', 'Al Buhayrah': 'Egypt', 'Al Iskandariyah': 'Egypt',
            'Al Isma’iliyah': 'Egypt', 'Al Jizah': 'Egypt', 'Aswan': 'Egypt', 'Asyut': 'Egypt',
            'Dumyat': 'Egypt', 'Janub Sina’': 'Egypt', 'Al Uqsur': 'Egypt', 'Faiyum': 'Egypt',

            # Jordan
            'Amman': 'Jordan',

            # Lebanon
            'Beirut': 'Lebanon',

            # Syria
            'As Suwayda': 'Syria', 'Latakia': 'Syria', 'Aleppo': 'Syria', 'Idlib': 'Syria',
            'Dara': 'Syria', 'Homs': 'Syria', 'Rif Dimashq': 'Syria',

            # Pakistan
            'Islamabad Capital Territory': 'Pakistan', 'Sindh': 'Pakistan', 'Balochistan': 'Pakistan',
            'North-West Frontier Province': 'Pakistan',

            # Bangladesh
            'Dhaka': 'Bangladesh', 'Sylhet': 'Bangladesh', 'Chittagong': 'Bangladesh', 'Rajshahi': 'Bangladesh',

            # Thailand
            'Chiang Rai': 'Thailand', 'Chaiyaphum': 'Thailand', 'Uttaradit': 'Thailand', 'Khon Kaen': 'Thailand',
            'Suphan Buri': 'Thailand', 'Krung Thep (Bangkok)': 'Thailand',

            # Laos
            'Champasak': 'Laos',

            # Vietnam
            'Ha Noi': 'Vietnam', 'Ho Chi Minh': 'Vietnam', 'Thanh Hoa': 'Vietnam',

            # Mozambique
            'Tete': 'Mozambique', 'Maputo (provincia)': 'Mozambique', 'Zambézia': 'Mozambique', 'Nampula': 'Mozambique',

            # Namibia
            'Erongo': 'Namibia', 'Otjozondjupa': 'Namibia', 'Khomas': 'Namibia', 'Karas': 'Namibia',

            # Botswana
            'Gaborone': 'Botswana',

            # Ghana
            'Volta': 'Ghana', 'Brong Ahafo': 'Ghana', 'Ashanti': 'Ghana', 'Central': 'Ghana',
            'Eastern': 'Ghana', 'Greater Accra': 'Ghana', 'Northern': 'Ghana', 'Western': 'Ghana',

            # Nigeria
            'Lagos': 'Nigeria',

            # Senegal
            'Dakar': 'Senegal',

            # Cameroon
            'Est': 'Cameroon', 'Littoral': 'Cameroon', 'Sud': 'Cameroon',

            # Gabon
            'Estuaire': 'Gabon',

            # Democratic Republic of the Congo
            'Bas-Congo': 'Democratic Republic of the Congo', 'Bandundu': 'Democratic Republic of the Congo',
            'Katanga': 'Democratic Republic of the Congo', 'Kinshasa': 'Democratic Republic of the Congo',
            'Orientale': 'Democratic Republic of the Congo', 'Equateur': 'Democratic Republic of the Congo',
            'Maniema': 'Democratic Republic of the Congo', 'Nord-Kivu': 'Democratic Republic of the Congo',

            # Angola
            'Luanda': 'Angola',

            # South Sudan
            'Juba': 'South Sudan',

            # Sudan
            'Khartoum': 'Sudan',

            # Madagascar
            'Mahajanga': 'Madagascar', 'Antananarivo': 'Madagascar', 'Toamasina': 'Madagascar',
            'Fianarantsoa': 'Madagascar', 'Toliara': 'Madagascar', 'Antsiranana': 'Madagascar',

            # India
            'Delhi': 'India', 'Goa': 'India', 'Kerala': 'India', 'Tamil Nadu': 'India', 'Maharashtra': 'India',
            'West Bengal': 'India', 'Assam': 'India', 'Jharkhand': 'India', 'Uttar Pradesh': 'India',
            'Himachal Pradesh': 'India', 'Karnataka': 'India', 'Andhra Pradesh': 'India', 'Madhya Pradesh': 'India',
            'Punjab': 'India', 'Bihar': 'India', 'Manipur': 'India', 'Sikkim': 'India',
            'Uttarakhand (Uttaranchal)': 'India', 'Meghalaya': 'India', 'Orissa': 'India', 'Rajasthan': 'India',
            'Gujarat': 'India', 'Kochi': 'India',

            # China
            'Hainan': 'China', 'Guangdong': 'China', 'Guangxi Zhuang': 'China', 'Yunnan': 'China',
            'Sichuan': 'China', 'Beijing': 'China', 'Shanghai': 'China', 'Jiangxi': 'China', 'Jiangsu': 'China',
            'Liaoning': 'China', 'Jilin': 'China', 'Heilongjiang': 'China', 'Hebei': 'China', 'Shandong': 'China',
            'Anhui': 'China', 'Hubei': 'China', 'Hunan': 'China', 'Fujian': 'China', 'Zhejiang': 'China',
            'Henan': 'China', 'Xinjiang Uygur': 'China', 'Neimenggu (Inner Mongolia)': 'China', 'Tibet': 'China',
            'Xizang (Tibet)': 'China', 'Aomen (Macau)': 'China', 'Xianggang (Hongkong)': 'China', 'Qinghai': 'China',
            'Shaanxi': 'China', 'Gansu': 'China', 'Guizhou': 'China', 'Shanxi': 'China',

            # Russia
            'Komi': 'Russia', 'Moskovskaya oblast': 'Russia', 'Amurskaya oblast': 'Russia', 'Irkutskaya oblast': 'Russia',
            'Rostovskaya oblast': 'Russia', 'Kamchatskiy kray': 'Russia', 'Primorskiy kray': 'Russia',
            'Arkhangelskaya oblast': 'Russia', 'Chechnya': 'Russia', 'Bashkortostan': 'Russia', 'Altay': 'Russia',
            'Leningradskaya oblast': 'Russia', 'Krasnoyarskiy kray': 'Russia', 'Volgogradskaya oblast': 'Russia',
            'Smolenskaya oblast': 'Russia', 'Dagestan': 'Russia', 'Novgorodskaya oblast': 'Russia',
            'Tyumenskaya oblast': 'Russia', 'Kurskskaya oblast': 'Russia', 'Kirovskaya oblast': 'Russia',
            'Kaliningradskaya oblast': 'Russia', 'Khabarovsk Krai': 'Russia', 'Karelia': 'Russia',
            'Sakhalinskaya oblast': 'Russia', 'Tatarstan': 'Russia', 'Udmurtiya': 'Russia', 'Krasnodarskiy kray': 'Russia',
            'Stavropolskiy kray': 'Russia', 'Vladimirskaya oblast': 'Russia', 'Yamala-Nenetskiy okrug': 'Russia',
            'Nenetskiy okrug': 'Russia', 'Severnaya Osetiya - Alaniya': 'Russia', 'Ulyanovskaya oblast': 'Russia',
            'Yaroslavskaya oblast': 'Russia', 'Sverdlovskaya oblast': 'Russia', 'Tambovskaya oblast': 'Russia',
            'Tomskaya oblast': 'Russia', 'Voronezhskaya oblast': 'Russia', 'Magadanskaya oblast': 'Russia',
            'Permskiy kray': 'Russia', 'Samarskaya oblast': 'Russia', 'Astrakhanskaya oblast': 'Russia',
            'Chukotka Autonomous Okrug': 'Russia', 'Chuvashia': 'Russia', 'Kalmykiya': 'Russia', 'Kemerovo Oblast': 'Russia',
            'Kostroma Oblast': 'Russia', 'Kurganskaya oblast': 'Russia', 'Lipetsk Oblast': 'Russia', 'Mari El': 'Russia',
            'Mordovia': 'Russia', 'Murmanskaya oblast': 'Russia', 'Orenburgskaya oblast': 'Russia', 'Penza Oblast': 'Russia',
            'Pskovskaya oblast': 'Russia', 'Ryazanskaya oblast': 'Russia', 'Saratovskaya oblast': 'Russia',
            'Tula Oblast': 'Russia', 'Tver Oblast': 'Russia', 'Vologda Oblast': 'Russia', 'Yamalo-Nenets Autonomous Okrug': 'Russia',
            'Orlovskaya oblast': 'Russia', 'Bryanskaya oblast': 'Russia', 'Omskaya oblast': 'Russia',
            'Kaluzhskaya oblast': 'Russia', 'Astrakhanskaya oblast': 'Russia', 'Sakhalinskaya oblast': 'Russia',
            'Tomskaya oblast': 'Russia', 'Samarskaya oblast': 'Russia', 'Stavropolskiy kray': 'Russia',
            'Krasnodarskiy kray': 'Russia', 'Krasnoyarskiy kray': 'Russia', 'Kamchatskiy kray': 'Russia',
            'Permskiy kray': 'Russia', 'Udmurtiya': 'Russia', 'Tatarstan': 'Russia', 'Kaliningradskaya oblast': 'Russia',
            'Leningradskaya oblast': 'Russia', 'Smolenskaya oblast': 'Russia', 'Volgogradskaya oblast': 'Russia',
            'Rostovskaya oblast': 'Russia', 'Irkutskaya oblast': 'Russia', 'Amurskaya oblast': 'Russia',
            'Nenetskiy okrug': 'Russia', 'Yamala-Nenetskiy okrug': 'Russia', 'Dagestan': 'Russia', 'Chechnya': 'Russia',
            'Bashkortostan': 'Russia', 'Altay': 'Russia', 'Komi': 'Russia', 'Karelia': 'Russia',
            'Severnaya Osetiya - Alaniya': 'Russia', 'Ulyanovskaya oblast': 'Russia', 'Yaroslavskaya oblast': 'Russia',
            'Sverdlovskaya oblast': 'Russia', 'Tambovskaya oblast': 'Russia', 'Voronezhskaya oblast': 'Russia',
            'Ryazanskaya oblast': 'Russia', 'Saratovskaya oblast': 'Russia', 'Pskovskaya oblast': 'Russia',
            'Vladimirskaya oblast': 'Russia', 'Vologda Oblast': 'Russia', 'Tula Oblast': 'Russia',
        
        def map_province_to_country(province, mapping):
            province_lower = str(province).lower()
            for key, country in mapping.items():
                if key.lower() in province_lower:
                    return country
            return 'Unknown'
        
        biodata['country'] = biodata['stateProvince_cleaned'].apply(
            lambda x: map_province_to_country(x, province_to_country))
        biodata['country'] = biodata['country'].fillna('Unknown')
        
        # Label encoding
        le_stateProvince = LabelEncoder()
        biodata['stateProvince_encoded'] = le_stateProvince.fit_transform(biodata['stateProvince_cleaned'])
        
        le_family = LabelEncoder()
        biodata['family_encoded'] = le_family.fit_transform(biodata['family_cleaned'])
        
        le_country = LabelEncoder()
        biodata['country_encoded'] = le_country.fit_transform(biodata['country'])
        
        return biodata, le_stateProvince, le_family, le_country
    
    def add_temp_humidity(self, biodata, tempdata, province_coordinates):
        """Add temperature and humidity data"""
        province_temp_hum = {}
        for province in biodata['stateProvince'].unique():
            if province not in province_coordinates:
                province_temp_hum[province] = (np.nan, np.nan)
            else:
                lat, lon = province_coordinates[province]
                tempdata['dist'] = np.sqrt((tempdata['latitude'] - lat)**2 + (tempdata['longitude'] - lon)**2)
                nearest = tempdata.loc[tempdata['dist'].idxmin()]
                province_temp_hum[province] = (nearest['temperature_C'], nearest['humidity'])
        
        biodata['temperature'] = biodata['stateProvince'].map(
            lambda x: province_temp_hum.get(x, (np.nan, np.nan))[0])
        biodata['humidity'] = biodata['stateProvince'].map(
            lambda x: province_temp_hum.get(x, (np.nan, np.nan))[1])
        
        return biodata
    
    def train_model(self, biodata, top_n=50):
        """Train the prediction model"""
        # Filter top genera
        top_genera = biodata['genus'].value_counts().nlargest(top_n).index
        biodata_top = biodata[biodata['genus'].isin(top_genera)].copy()
        
        le_genus = LabelEncoder()
        biodata_top['genus_encoded'] = le_genus.fit_transform(biodata_top['genus'])
        
        # Feature engineering
        biodata_top['temp_squared'] = biodata_top['temperature'] ** 2
        global_mean_temp = biodata_top['temperature'].mean()
        biodata_top['temp_diff_mean'] = biodata_top['temperature'] - global_mean_temp
        biodata_top['temp_family_interaction'] = biodata_top['temperature'] * biodata_top['family_encoded']
        biodata_top['humidity_squared'] = biodata_top['humidity'] ** 2
        biodata_top['humidity_temp_interaction'] = biodata_top['humidity'] * biodata_top['temperature']
        
        features = [
            'stateProvince_encoded', 'family_encoded', 'country_encoded', 'temperature', 'humidity',
            'temp_squared', 'temp_diff_mean', 'temp_family_interaction', 'humidity_squared', 'humidity_temp_interaction'
        ]
        
        X = biodata_top[features].fillna(-1)
        y = biodata_top['genus_encoded']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
        model.fit(X_scaled, y)
        
        y_pred = model.predict(X_scaled)
        acc = accuracy_score(y, y_pred)
        bal_acc = balanced_accuracy_score(y, y_pred)
        
        return model, scaler, le_genus, features, global_mean_temp, biodata_top, acc, bal_acc
    
    def refresh_data(self):
        """Refresh the data display"""
        if self.biodata is not None:
            # Clear existing data
            for item in self.data_tree.get_children():
                self.data_tree.delete(item)
            
            # Set up columns
            self.data_tree['columns'] = list(self.biodata.columns)
            self.data_tree['show'] = 'headings'
            
            for col in self.data_tree['columns']:
                self.data_tree.heading(col, text=col)
                self.data_tree.column(col, width=100)
            
            # Add data
            for idx, row in self.biodata.head(1000).iterrows():
                self.data_tree.insert('', 'end', values=list(row))
    
    def export_data(self):
        """Export data to CSV"""
        if self.biodata is not None:
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
            if filename:
                self.biodata.to_csv(filename, index=False)
                messagebox.showinfo("Success", f"Data exported to {filename}")
    
    def make_prediction(self):
        """Make genus predictions"""
        if self.final_model is None:
            messagebox.showwarning("Warning", "Model not loaded yet. Please wait for data to load.")
            return
        
        country = self.country_var.get()
        province = self.province_var.get()
        family = self.family_var.get()
        top_k = int(self.top_k_var.get())
        
        if not province:
            messagebox.showerror("Error", "Please enter a province/state name")
            return
        
        try:
            # Get families for this province
            all_families_prov = self.biodata[
                self.biodata['stateProvince_cleaned'] == province]['family_cleaned'].unique().tolist()
            
            # Make prediction
            df_predictions = self.predict_genera(province, country, top_k, 
                                             family_names=all_families_prov if not family else [family])
            
            # Clear existing results
            for item in self.results_tree.get_children():
                self.results_tree.delete(item)
            
            # Display results
            for idx, row in df_predictions.iterrows():
                self.results_tree.insert('', 'end', values=(
                    idx + 1,
                    row['Genus'],
                    f"{row['Probability']:.2%}"
                ))
                
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
    
    def predict_genera(self, province_name, country_name=None, top_k=10, family_names=None):
        """Predict top genera for given parameters"""
        if family_names is None or len(family_names) == 0:
            family_names = ['Unknown']
        
        # Encode inputs
        if province_name in self.le_stateProvince.classes_:
            province_enc = self.le_stateProvince.transform([province_name])[0]
        else:
            province_enc = -1
        
        if not country_name:
            country_name = province_name
        if country_name in self.le_country.classes_:
            country_enc = self.le_country.transform([country_name])[0]
        else:
            country_enc = -1
        
        # Get temperature and humidity
        def get_temp_humidity_for_province(province):
            if province not in province_coordinates:
                return np.nan, np.nan
            lat, lon = province_coordinates[province]
            self.tempdata['dist'] = np.sqrt(
                (self.tempdata['latitude'] - lat)**2 + (self.tempdata['longitude'] - lon)**2)
            nearest = self.tempdata.loc[self.tempdata['dist'].idxmin()]
            return nearest['temperature_C'], nearest['humidity']
        
        temp, humidity = get_temp_humidity_for_province(province_name)
        
        # Calculate features
        temp_squared = temp ** 2 if not np.isnan(temp) else -1
        temp_diff_mean = temp - self.global_mean_temp if not np.isnan(temp) else -1
        humidity_squared = humidity ** 2 if not np.isnan(humidity) else -1
        humidity_temp_interaction = humidity * temp if not (np.isnan(humidity) or np.isnan(temp)) else -1
        
        # Make predictions for each family
        probas = []
        for family_encoded in [self.le_family.transform([fam])[0] if fam in self.le_family.classes_ else 0 
                              for fam in family_names]:
            temp_family_interaction = temp * family_encoded if not np.isnan(temp) else -1
            
            input_df = pd.DataFrame([[
                province_enc, family_encoded, country_enc, temp, humidity,
                temp_squared, temp_diff_mean, temp_family_interaction,
                humidity_squared, humidity_temp_interaction
            ]], columns=self.features)
            
            input_scaled = self.scaler.transform(input_df)
            proba = self.final_model.predict_proba(input_scaled)[0]
            probas.append(proba)
        
        # Average probabilities
        avg_proba = np.mean(probas, axis=0)
        top_indices = np.argsort(avg_proba)[-top_k:][::-1]
        top_proba = avg_proba[top_indices]
        top_genera_pred = self.le_genus.inverse_transform(top_indices)
        
        return pd.DataFrame({
            'Genus': top_genera_pred,
            'Probability': top_proba,
            'InputProvince': province_name
        })

def main():
    """Main application entry point"""
    root = tk.Tk()
    app = MushroomTrackerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
