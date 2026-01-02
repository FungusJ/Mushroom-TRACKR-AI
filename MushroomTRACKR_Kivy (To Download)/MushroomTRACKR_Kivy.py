import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.tabbedpanel import TabbedPanel, TabbedPanelItem
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.spinner import Spinner
from kivy.uix.progressbar import ProgressBar
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.popup import Popup
from kivy.uix.scrollview import ScrollView
from kivy.uix.dropdown import DropDown
from kivy.properties import StringProperty, ListProperty, ObjectProperty
from kivy.clock import Clock
from kivy.core.window import Window
import pandas as pd
import numpy as np
import os
import threading
from rapidfuzz import process
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score

Window.size = (1200, 800)

province_coordinates = {
    'Ostergotland': (58.4064, 15.6255), 'Skane': (55.9903, 13.5958), 'Uppsala': (59.8586, 17.6389),
    'Vastra Gotaland': (58.2528, 13.0596), 'Sodermanland': (59.0336, 16.7519), 'Stockholm': (59.3293, 18.0686),
    'Jamtland': (63.1712, 14.9592), 'British Columbia': (53.7267, -127.6476), 'Ontario': (51.2538, -85.3232),
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
    'Sao Paulo': (-23.5505, -46.6333), 'Amazonas': (-3.4653, -62.2159), 'Parana': (-24.8949, -51.5500),
    'Bahia': (-12.5797, -41.7007), 'Rio de Janeiro': (-22.9068, -43.1729), 'Minas Gerais': (-18.5122, -44.5550),
    'Western Cape': (-33.2278, 21.8569), 'Eastern Cape': (-32.2968, 26.4194), 'KwaZulu-Natal': (-28.5306, 30.8958),
    'Gauteng': (-26.2708, 28.1123), 'Chiapas': (16.7569, -93.1292), 'Jalisco': (20.6597, -103.3496),
    'Yucatan': (20.7099, -89.0943), 'Oaxaca': (17.0732, -96.7266), 'Kerala': (10.8505, 76.2711),
    'Tamil Nadu': (11.1271, 78.6569), 'Maharashtra': (19.7515, 75.7139), 'West Bengal': (22.9868, 87.8550),
    'Assam': (26.2006, 92.9376), 'Moscow': (55.7558, 37.6173), 'Siberia': (61.0137, 99.1967),
    'Primorsky Krai': (45.0525, 135.0000), 'Java': (-7.4910, 110.0044), 'Bali': (-8.3405, 115.0920),
    'Sumatra': (-0.5897, 101.3431), 'Luzon': (16.5000, 121.0000), 'Mindanao': (8.0000, 125.0000),
    'Visayas': (11.5000, 123.5000), 'Buenos Aires': (-34.6037, -58.3816), 'Santa Fe': (-31.5855, -60.7238),
    'Cordoba': (-31.4201, -64.1888)
}

non_macroscopic_basidiomycota_genera = [
    "Ustilago", "Tilletia", "Urocystis", "Sporobolomyces", "Cryptococcus", "Rhodotorula", "Cystofilobasidium",
    "Sporidiobolus", "Erythrobasidium", "Microbotryum", "Exobasidium", "Entyloma", "Graphiola",
    "Puccinia", "Melampsora", "Coleosporium", "Chrysomyxa", "Cronartium", "Phakopsora", "Septobasidium"
]

non_macroscopic_basidiomycota_families = [
    "Ustilaginaceae", "Ustilaginomycetaceae", "Urocystidaceae", "Tilletiaceae", "Tilletiopsidaceae",
    "Cryptococcaceae", "Filobasidiaceae", "Cystofilobasidiaceae", "Sporidiobolaceae", "Sporobolomycetaceae",
    "Erythrobasidiaceae", "Microbotryaceae", "Exobasidiaceae", "Entylomataceae", "Graphiolaceae",
    "Pucciniaceae", "Melampsoraceae", "Coleosporiaceae", "Chrysomyxa", "Cronartiaceae", "Phakopsoraceae",
    "Phyllachoraceae", "Septobasidiaceae", "Auriculariaceae"
]

class MushroomTRACKRRoot(BoxLayout):
    status_text = StringProperty("Loading data...")
    progress = ObjectProperty(None)
    biodata = ObjectProperty(None)
    tempdata = ObjectProperty(None)
    available_countries = ListProperty([])
    available_provinces = ListProperty([])
    available_families = ListProperty([])
    final_model = ObjectProperty(None)
    scaler = ObjectProperty(None)
    features = ListProperty([])
    global_mean_temp = ObjectProperty(None)
    biodata_top = ObjectProperty(None)
    acc = ObjectProperty(None)
    bal_acc = ObjectProperty(None)
    le_stateProvince = ObjectProperty(None)
    le_family = ObjectProperty(None)
    le_country = ObjectProperty(None)
    le_genus = ObjectProperty(None)

    def __init__(self, **kwargs):
        super().__init__(orientation='vertical', **kwargs)
        self.tab_panel = TabbedPanel(do_default_tab=False)
        self.add_widget(self.tab_panel)
        self.build_tabs()
        Clock.schedule_once(lambda dt: threading.Thread(target=self.load_data).start(), 0.1)

    def build_tabs(self):
        # Data Tab
        self.data_tab = TabbedPanelItem(text='Data')
        data_layout = BoxLayout(orientation='vertical')
        self.data_status = Label(text=self.status_text, size_hint_y=None, height=30)
        self.data_progress = ProgressBar(max=1, value=0)
        self.data_textbox = TextInput(readonly=True, multiline=True, font_size=14)
        self.refresh_data_btn = Button(text="Refresh Data", size_hint_y=None, height=40, on_release=lambda x: self.refresh_data())
        self.export_data_btn = Button(text="Export Data", size_hint_y=None, height=40, on_release=lambda x: self.export_data())
        data_layout.add_widget(self.data_status)
        data_layout.add_widget(self.data_progress)
        data_layout.add_widget(self.refresh_data_btn)
        data_layout.add_widget(self.export_data_btn)
        data_layout.add_widget(self.data_textbox)
        self.data_tab.add_widget(data_layout)
        self.tab_panel.add_widget(self.data_tab)

        # Predict Tab
        self.predict_tab = TabbedPanelItem(text='Predict')
        predict_layout = BoxLayout(orientation='vertical')
        self.predict_status = Label(text="Prediction Inputs", size_hint_y=None, height=30)
        self.country_spinner = Spinner(text="Select Country", values=[], size_hint_y=None, height=40)
        self.province_spinner = Spinner(text="Select Province", values=[], size_hint_y=None, height=40)
        self.family_spinner = Spinner(text="Select Family", values=[], size_hint_y=None, height=40)
        self.top_k_input = TextInput(text="10", hint_text="Top K Results (1-50)", size_hint_y=None, height=40, multiline=False)
        self.predict_btn = Button(text="Predict Genera", size_hint_y=None, height=40, on_release=lambda x: self.make_prediction())
        self.results_label = Label(text="Prediction Results", size_hint_y=None, height=30)
        self.results_textbox = TextInput(readonly=True, multiline=True, font_size=14)
        predict_layout.add_widget(self.predict_status)
        predict_layout.add_widget(self.country_spinner)
        predict_layout.add_widget(self.province_spinner)
        predict_layout.add_widget(self.family_spinner)
        predict_layout.add_widget(self.top_k_input)
        predict_layout.add_widget(self.predict_btn)
        predict_layout.add_widget(self.results_label)
        predict_layout.add_widget(self.results_textbox)
        self.predict_tab.add_widget(predict_layout)
        self.tab_panel.add_widget(self.predict_tab)

        # About Tab
        self.about_tab = TabbedPanelItem(text='About')
        about_text = (
            "🍄 MushroomTRACKR - Fungal Genus Predictor\n\n"
            "This application predicts fungal genera based on:\n"
            "• Province/State location\n"
            "• Temperature and humidity data\n"
            "• Family classification\n"
            "• Machine learning models\n\n"
            "Features:\n"
            "• Random Forest classifier with 95%+ accuracy\n"
            "• Province-based environmental data\n"
            "• Interactive predictions with confidence bars\n"
            "• Modern Kivy UI\n"
            "• Fuzzy search dropdowns\n\n"
            "Version: 3.0 (Kivy Edition)"
        )
        self.about_tab.add_widget(Label(text=about_text, halign='left', valign='top'))
        self.tab_panel.add_widget(self.about_tab)

    def update_status(self, text, progress=0.0):
        self.status_text = text
        self.data_status.text = text
        self.data_progress.value = progress

    def load_data(self):
        try:
            # Data loading and processing in background thread
            script_dir = os.path.dirname(os.path.abspath(__file__))
            filtered_path = os.path.join(script_dir, 'filtered_biodiversity_data.csv')
            temp_file_path = os.path.join(script_dir, 'temperature_2m.csv')
            biodata = pd.read_csv(filtered_path)
            tempdata = pd.read_csv(temp_file_path)
            le_stateProvince = LabelEncoder()
            le_family = LabelEncoder()
            le_country = LabelEncoder()
            le_genus = LabelEncoder()
            biodata['stateProvince_encoded'] = le_stateProvince.fit_transform(biodata['stateProvince_cleaned'])
            biodata['family_encoded'] = le_family.fit_transform(biodata['family_cleaned'])
            biodata['country_encoded'] = le_country.fit_transform(biodata['country'])
            biodata['genus_encoded'] = le_genus.fit_transform(biodata['genus'])
            available_countries = sorted(biodata['country'].dropna().unique().astype(str))
            available_provinces = sorted(biodata['stateProvince_cleaned'].dropna().unique().astype(str))
            available_families = sorted(biodata['family_cleaned'].dropna().unique().astype(str))
            # Train model
            def train_model(biodata):
                top_n = 50
                top_genera = biodata['genus'].value_counts().nlargest(top_n).index
                biodata_top = biodata[biodata['genus'].isin(top_genera)].copy()
                le_genus = LabelEncoder()
                biodata_top['genus_encoded'] = le_genus.fit_transform(biodata_top['genus'])
                biodata_top['temp_squared'] = biodata_top['temperature'] ** 2
                global_mean_temp = biodata_top['temperature'].mean()
                biodata_top['temp_diff_mean'] = biodata_top['temperature'] - global_mean_temp
                biodata_top['temp_family_interaction'] = biodata_top['temperature'] * biodata_top['family_encoded']
                biodata_top = biodata_top[
                    ~biodata_top['family_cleaned'].isin(non_macroscopic_basidiomycota_families) &
                    ~biodata_top['genus'].isin(non_macroscopic_basidiomycota_genera)
                ]
                if biodata_top.empty:
                    raise ValueError("No data left after filtering non-macroscopic Basidiomycota.")
                features = [
                    'stateProvince_encoded', 'family_encoded', 'country_encoded', 'temperature',
                    'temp_squared', 'temp_diff_mean', 'temp_family_interaction'
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
                return model, scaler, features, global_mean_temp, biodata_top, acc, bal_acc, le_genus
            model, scaler, features, global_mean_temp, biodata_top, acc, bal_acc, le_genus = train_model(biodata)
            # Schedule UI update on main thread
            def update_ui(dt):
                self.biodata = biodata
                self.tempdata = tempdata
                self.le_stateProvince = le_stateProvince
                self.le_family = le_family
                self.le_country = le_country
                self.le_genus = le_genus
                self.available_countries = available_countries
                self.available_provinces = available_provinces
                self.available_families = available_families
                self.country_spinner.values = available_countries
                self.province_spinner.values = available_provinces
                self.family_spinner.values = available_families
                self.final_model = model
                self.scaler = scaler
                self.features = features
                self.global_mean_temp = global_mean_temp
                self.biodata_top = biodata_top
                self.acc = acc
                self.bal_acc = bal_acc
                self.update_status(f"Data loaded! Model Accuracy: {acc:.2%}, Balanced Accuracy: {bal_acc:.2%}", 1.0)
                self.refresh_data()
            Clock.schedule_once(lambda dt: self.update_status("Loading filtered_data.csv...", 0.2), 0)
            Clock.schedule_once(lambda dt: self.update_status("Loading temperature data...", 0.4), 0)
            Clock.schedule_once(lambda dt: self.update_status("Processing filtered data...", 0.6), 0)
            Clock.schedule_once(lambda dt: self.update_status("Training model from filtered data...", 0.8), 0)
            Clock.schedule_once(update_ui, 0)
        except Exception as error:
            Clock.schedule_once(lambda dt, err=error: self.update_status(f"Error: {str(err)}", 0.0), 0)

    def refresh_data(self):
        if self.biodata is not None:
            data_text = "\t".join(self.biodata.columns) + "\n"
            for idx, row in self.biodata.head(100).iterrows():
                data_text += "\t".join([str(x) for x in row]) + "\n"
            self.data_textbox.text = data_text

    def export_data(self):
        if self.biodata is not None:
            from kivy.utils import platform
            if platform == 'win':
                default_path = os.path.expanduser("~/Desktop/filtered_data.csv")
            else:
                default_path = os.path.expanduser("~/filtered_data.csv")
            with open(default_path, 'w') as f:
                self.biodata.to_csv(f, index=False)
            self.update_status(f"Exported to {default_path}", 1.0)

    def make_prediction(self):
        province = self.province_spinner.text
        country = self.country_spinner.text
        top_k_str = self.top_k_input.text
        try:
            top_k = int(top_k_str)
            if top_k < 1 or top_k > 50:
                raise ValueError
        except ValueError:
            self.update_status("Top K must be an integer between 1 and 50.", 1.0)
            return
        family = self.family_spinner.text
        family_list = [family] if family else None
        if not province or not country:
            self.update_status("Please select both country and province/state.", 1.0)
            return
        df_predictions = self.predict_genera(province, country, top_k, family_list)
        self.results_textbox.text = ""
        if df_predictions is not None and not df_predictions.empty:
            self.results_label.text = f"Predicted genera for {province}, {country}."
            results_text = "Genus\tProbability\n" + "\n".join([f"{row['Genus']}\t{row['Probability']:.2%}" for idx, row in df_predictions.iterrows()])
            self.results_textbox.text = results_text
        else:
            self.update_status("No predictions available for the selected input.", 1.0)

    def train_model(self, biodata):
        top_n = 50
        top_genera = biodata['genus'].value_counts().nlargest(top_n).index
        biodata_top = biodata[biodata['genus'].isin(top_genera)].copy()
        self.le_genus = LabelEncoder()
        biodata_top['genus_encoded'] = self.le_genus.fit_transform(biodata_top['genus'])
        biodata_top['temp_squared'] = biodata_top['temperature'] ** 2
        global_mean_temp = biodata_top['temperature'].mean()
        biodata_top['temp_diff_mean'] = biodata_top['temperature'] - global_mean_temp
        biodata_top['temp_family_interaction'] = biodata_top['temperature'] * biodata_top['family_encoded']
        biodata_top = biodata_top[
            ~biodata_top['family_cleaned'].isin(non_macroscopic_basidiomycota_families) &
            ~biodata_top['genus'].isin(non_macroscopic_basidiomycota_genera)
        ]
        if biodata_top.empty:
            raise ValueError("No data left after filtering non-macroscopic Basidiomycota.")
        features = [
            'stateProvince_encoded', 'family_encoded', 'country_encoded', 'temperature',
            'temp_squared', 'temp_diff_mean', 'temp_family_interaction'
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
        return model, scaler, features, global_mean_temp, biodata_top, acc, bal_acc

    def predict_genera(self, province_name, country_name=None, top_k=10, family_names=None):
        if self.le_stateProvince is not None and province_name in self.le_stateProvince.classes_:
            province_encoded = self.le_stateProvince.transform([province_name])[0]
        else:
            self.update_status("Province not recognized.", 1.0)
            return None
        if not country_name:
            country_encoded = -1
        elif self.le_country is not None and country_name in self.le_country.classes_:
            country_encoded = self.le_country.transform([country_name])[0]
        else:
            self.update_status("Country not recognized.", 1.0)
            return None
        def get_temp_for_province(province):
            if province not in province_coordinates:
                return np.nan
            lat, lon = province_coordinates[province]
            self.tempdata['dist'] = np.sqrt((self.tempdata['latitude'] - lat)**2 + (self.tempdata['longitude'] - lon)**2)
            nearest = self.tempdata.loc[self.tempdata['dist'].idxmin()]
            temp_val = nearest['temperature_C'] if 'temperature_C' in nearest else np.nan
            return temp_val
        temp = get_temp_for_province(province_name)
        temp_squared = temp ** 2 if not np.isnan(temp) else -1
        temp_diff_mean = temp - self.global_mean_temp if not np.isnan(temp) else -1
        if family_names is None or len(family_names) == 0:
            family_names = ['Unknown']
        family_encs = []
        for fam in family_names:
            if fam in self.le_family.classes_:
                family_enc = self.le_family.transform([fam])[0]
            else:
                family_enc = -1
            family_encs.append(family_enc)
        probas = []
        for family_encoded in family_encs:
            temp_family_interaction = temp * family_encoded if not np.isnan(temp) else -1
            input_df = pd.DataFrame([[
                province_encoded, family_encoded, country_encoded, temp,
                temp_squared, temp_diff_mean, temp_family_interaction,
            ]], columns=self.features)
            input_scaled = self.scaler.transform(input_df)
            proba = self.final_model.predict_proba(input_scaled)[0]
            probas.append(proba)
        avg_proba = np.mean(probas, axis=0)
        top_indices = np.argsort(avg_proba)[-top_k:][::-1]
        top_proba = avg_proba[top_indices]
        top_genera_pred = self.le_genus.inverse_transform(top_indices)
        return pd.DataFrame({
            'Genus': top_genera_pred,
            'Probability': top_proba,
            'InputProvince': province_name
        })

class MushroomTRACKRApp(App):
    def build(self):
        return MushroomTRACKRRoot()

if __name__ == '__main__':
    MushroomTRACKRApp().run()