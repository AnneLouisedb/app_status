# import external packages
import spacy
from spacy.tokens import Doc, Span
from spacy.language import Language
from spacy import displacy
#import crosslingual_coreference
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import re
import textacy
import pandas as pd
import numpy as np
import plotly.express as px
import requests
import country_converter as coco
from geopy.geocoders import Nominatim
from newspaper import Article
import streamlit as st

import os

# Get the directory path of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Specify the folder name
folder_name = "models/SKC_model_new"

# Construct the full path to the folder
folder_path = os.path.join(current_dir, folder_name)

def is_web_link(text):
    pattern = r'^https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
    return re.match(pattern, text) is not None

@Language.component("set_custom_weapons")
def set_custom_weapons(doc):
    entities = list(doc.ents)  # Convert doc.ents to a list

    for token in doc:

        if any(keyword in token.text.lower() for keyword in ['weapon', 'pistol', 'firearm', "arms"]):
            token.ent_type_ = 'WEAPON'
            entities.append(Span(doc, token.i, token.i + 1, label="WEAPON"))  # Create a new Span object for the weapon entity

        for i in range(len(doc) - 1):
            if doc[i].text.lower() == 'weapons' and doc[i + 1].text.lower() == 'trafficking':
                entities.append(Span(doc, doc[i].i, doc[i + 2].i + 1, label="ILLEGAL_ACTIVITY"))  # Create a new Span object for the illegal activity entity
        
    
            
            #if token.text.lower() == 'trafficking' or token.text.lower() == 'weapons trafficking':
        #    entities.append(Span(doc, token.i, token.i + 1, label="ILLEGAL_ACTIVITY"))  # Create a new Span object for the illegal activity entity

    # Remove any conflicting spans
    entities = spacy.util.filter_spans(entities)
    doc.ents = entities  # Assign the updated list to doc.ents

    return doc

@spacy.Language.component("set_custom_norp")
def set_custom_norplabel(doc):
    entities = list(doc.ents)  
    spans = []
    for token in doc:
        if token.ent_type_ == "NORP":
            # Find the head noun of the NORP token
            noun = token.head
            while noun.pos_ != "NOUN":
                noun = noun.head

            # Create a new span that covers both the NORP and the corresponding noun
            start = min(token.i, noun.i)
            end = max(token.i, noun.i) + 1
            span = doc[start:end]
            span_label = token.ent_type_ + "_" + "NOUN"  # Combine the entity types
            span = spacy.tokens.Span(doc, span.start, span.end, label=span_label)
            
            # Add the merged span to the list of spans
            spans.append(span)
       
    # Remove any conflicting spans
    entities = spacy.util.filter_spans(entities + spans)
    doc.ents = entities  # Assign the updated list to doc.ents

    return doc


class knowledge_base_NLP():
    """ The knowledge base should contain the information from the domain experts. The column titles should include 
    Source, Date, Location, Weapons.
    First we build a knowledge base by importing the text by a domain expert and extracting weapons, locations and relations. 
    1. we call the coref_document
    2. coreferencing of the document, store the sentences parsed per sentence
    3. relation extraction on these (coreferences sentences) - using textacy"""

    def __init__(self, input_text, lang = "en") -> None:
        self.language = lang 
        self.DEVICE = -1
        self.raw_text = input_text
        
        coref_doc = self.coref_document()
        self.coref_doc_original = coref_doc
        self.coreffed_sents = list(coref_doc.sents)
        self.coref_docs = self.coref_docs()
        self.entities = coref_doc.ents
        
        self.attributes = ["LOC","GPE", "DATE", "ORG"] # location and date
        self.pos_relations = self.NER_frame()
        self.extract_attributes = self.extracting_location_weapon()

    def pretrained_model(self):
        #nlp = spacy.load('en_core_web_sm') # this should be the pretrained model from issue #2
        # Add the custom component to the pipeline

        base_model = spacy.load("en_core_web_sm")
        # Load the spaCy model from the hosted location
        #model_path = "https://drive.google.com/drive/folders/1YtvgVSmriItX2DYQcM1hPv-YOCnQavtO?usp=sharing"
        #ner_weapons = spacy.load(model_path)

        # loading the pre-trained model that detects weapon-type entity labels
        ner_weapons = spacy.load(folder_path)

        base_model.add_pipe("ner", name="ner_weapons", source=ner_weapons)

        return base_model
    
    # Define a custom rule-based component for tagging "firearm" as "WEAPON"
    
    def kb_from_article(self, url):
        article = Article(url)
        article.download()
        article.parse()

        config = {
            "article_title": article.title,
            "article_publish_date": article.publish_date
        }
        return article.text

  
            
    def coref_document(self):
        """
        Coreference resolution is the task of determining when two or more expressions in a text refer to the same entity. 
        By enabling coreference resolution, the model will be able to understand and resolve pronouns, 
        definite noun phrases, and other referring expressions to their corresponding entities.
        """
        coref = spacy.load('en_core_web_sm', disable=['ner', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer'])
        #coref.add_pipe("xx_coref", config={"chunk_size": 2500, "chunk_overlap": 2, "device": self.DEVICE})
        
        # check if the input text is a web link
        if is_web_link(self.raw_text):
            raw_text = self.kb_from_article(self.raw_text)
        else:
            raw_text = self.raw_text
    
        #coref_text = coref(raw_text)._.resolved_text
        coref_text = raw_text

        nlp = self.pretrained_model()  # Load the pretrained model

        out = nlp(coref_text)  # Process the text using the pretrained model

        return out
    
    def coref_docs(self):
        """The function returns a list of sentences"""
        lst_docs = [sent for sent in self.coreffed_sents]
        print("tot sentences:", len(lst_docs))

        return lst_docs
      
    def POS_DEP(self, i):
        # i denotes the sentence number. 
        for token in self.coreffed_sents[i]:
            print(token.text, "-->", "pos: "+token.pos_, "|", "dep: "+token.dep_, "")

        displacy.render(self.coreffed_sents[i], style="dep", options={"distance":100})
        for tag in (self.coreffed_sents[i].ents):
            print(tag.text, f"({tag.label_})")

        displacy.render(self.coreffed_sents[i], style="ent")
        return ""
    
    def extracting_location_weapon(self):
        dic = {"id":[], "text":[], "weapons": [], "locations": [], "organisation": [], "date": [], 'person':[], 'weapon_adj': [], 'NORP noun':[]}

        for n,sentence in enumerate(self.coref_docs):
           
            dic["id"].append(n)
            dic["text"].append(sentence.text)
            locations = []
            dates = []
            weapons = []
            person = []
            weapon_adj = []
            orgs = []
            norp = []
            norpnoun = []

            for ent in sentence.ents:
                if ent.label_ == "PERSON":
                    person.append(ent.text)
                if ent.label_ == "Weapons":
                    weapons.append(ent.text)
                elif ent.label_ == "GPE":
                    locations.append(ent.text)
                elif ent.label_ == "DATE":
                    dates.append(ent.text)
                elif ent.label == "ORG":
                    orgs.append(ent.text)
                elif ent.label == "NORP":
                    norp.append(ent.text)
                elif ent.label == "NORP_NOUN":
                    norpnoun.append(ent.text)

            dic["weapons"].append(set(weapons))
            dic["locations"].append(set(locations))
            dic["date"].append(set(dates))
            dic["person"].append(set(person))
            dic['organisation'].append(set(orgs))
            dic['weapon_adj'].append(set(weapon_adj))
            dic['NORP noun'].append(set(norpnoun))

        dtf = pd.DataFrame(dic)
        # Drop rows with empty sets in all columns except for 'id' and 'text'
        cols_to_check = dtf.columns.difference(['id', 'text'])
        dtf = dtf.dropna(subset=cols_to_check, how='all')
        weapon_adjectives = []

        for index, row in dtf.iterrows():
            weapon = row['weapons']
            sentence_id = row['id']

            if pd.notna(weapon):
                sentence = self.coref_docs[sentence_id]
                for token in sentence:
                    if token.ent_type_ == "Weapons":
                        adjectives = [child.text for child in token.children if child.pos_ == "ADJ"]
                        if adjectives:
                            weapon_adjectives.append((adjectives, token.text))
                            # Write the list of adjectives to the 'weapon_adj' column in the corresponding row
                            dtf.at[index, 'weapon_adj'] = adjectives

        return dtf
    
    
    def NER_frame(self):
        dic = {
            "id": [],
            "text": [],
            "entity": [],
            "relation": [],
            "object": [],
            "object label": [],
            "subject label": [],
            "complete object": [],
            "complete subject": [],
            "relation_date_obj": [],
            "relation_date_subj": []}

        # Load the English model
        nlp = spacy.load('en_core_web_sm')

        for n,sentence in enumerate(self.coreffed_sents):
            lst_generators = list(textacy.extract.subject_verb_object_triples(sentence)) 
            for sent in lst_generators:
                subj = "_".join(map(str, sent.subject))
                obj  = "_".join(map(str, sent.object))
                relation = "_".join(map(str, sent.verb))
                dic["id"].append(n)
                dic["text"].append(sentence.text)
                dic["entity"].append(subj)
                dic["object"].append(obj)
                dic["relation"].append(relation)

                # Extract NER labels for subject and object
                doc = nlp(sentence.text)  # Process the sentence with spaCy
                
                # Extract NER labels for subject and object
                subj_labels = [token.ent_type_ for token in sent.subject if token.ent_type_]
                obj_labels = [token.ent_type_ for token in sent.object if token.ent_type_]
                dic["subject label"].append(subj_labels) # these labels will have to come from the pretrained model
                dic["object label"].append(obj_labels)  # these labels will have to come from the pretrained model

                # Find the complete object using POS tags
                complete_object = ""
                object_token = None  # Initialize object_token outside the loop
                relation_obj = None
                relation_date_obj = None

                for token in doc:
                    if token.text == obj:
                        object_token = token
                        break

                if object_token:
                    for child in object_token.children:
                        if child.dep_ == "prep":
                            prep_token = child
                            prep_children = [grandchild for grandchild in prep_token.children if grandchild.dep_ == "pobj"]
                            if prep_children:
                                relation = (object_token.text, prep_token.text, " ".join([grandchild.text for grandchild in prep_children[0].subtree]))
                                complete_object = relation
                            elif child.ent_type_ == "DATE":
                                relation_date_obj = child.text

                dic["complete object"].append(complete_object)
                dic["relation_date_obj"].append(relation_date_obj)

                # Find the complete subject using POS tags
                complete_subject = ""
                subject_token = None  # Initialize subject_token outside the loop
                relation_subj = None
                relation_date_subj = None

                for token in doc:
                    if token.text == subj:
                        subject_token = token
                        break

                if subject_token:
                    for child in subject_token.children:
                        if child.dep_ == "prep":
                            prep_token = child
                            prep_children = [grandchild for grandchild in prep_token.children if grandchild.dep_ == "pobj"]
                            if prep_children:
                                relation = (subject_token.text, prep_token.text, " ".join([grandchild.text for grandchild in prep_children[0].subtree]))
                                complete_subject = relation
                            elif child.ent_type_ == "DATE":
                                relation_date_subj = child.text
                                print(relation_date_subj)

                dic["complete subject"].append(complete_subject)
                dic["relation_date_subj"].append(relation_date_subj)

               
        dtf = pd.DataFrame(dic).astype(str)
        

        return dtf

    def chloropleth_weapon_locations(self):
        
        df = self.extracted_attributes.copy()
        df['ISO_A3'] = None
        df['Country_Location'] = None
        
        geolocator = Nominatim(user_agent="weapon_locations")
        # initialize country converter
        cc = coco.CountryConverter()

        # Get countries from URL
        worldmap = requests.get("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson").json()

        countries = []
        iso3_codes = []
        weapon_ids = []
        
        df = df.dropna(subset=['locations'])

        for index, row in df.iterrows():
            row['locations']= set(row['locations'])

        df['locations'] = df['locations'].apply(lambda x: list(x) if x is not None else None)
        df = df.explode('locations')

        df['weapons'] = df['weapons'].apply(lambda x: list(x) if x is not None else None)
        df = df.explode('weapons')


        for index, row in df.iterrows():
            location = row['locations']
            try:
                iso3_codes.append(cc.convert(names=str(location), to='ISO3'))
                countries.append(location)
            except:
                countries.append(None)
                iso3_codes.append(None)

        
            row['Country_Location'] = countries
            row['ISO_A3'] = iso3_codes
            

        df = df.explode('ISO_A3')  
        df['Country_Location'] = df['Country_Location'].apply(lambda x: list(x) if x is not None else None)
        df = df.explode('Country_Location')  #- this does not work because we need to explode a set type
        df['weapons'] = 5 #CHANGE THIS, this is a text value
        print(df)
    
        figmap = px.choropleth_mapbox(
                    df,
                    locations="ISO_A3", # Based on this ID!
                    geojson=worldmap,
                    featureidkey="properties.ADM0_A3",
                    color='weapons', # This is coloured!
                    color_continuous_scale='Blues',
                    hover_data=['weapons'],
                    #animation_frame="date",
                    range_color=[20, 10],
                    title="Weapons Timeline Pilot",
                    mapbox_style="carto-positron",
                    center={"lat": 54.5260, "lon": 15.2551}
                )
        return figmap
    



    # begin a streamlit app
def app():
    st.write(folder_path)
    """
    Function that creates the page, which is subdivided into horizontal boxes.
    """
    st.title('Monitor demo')
    user_input = st.text_input("Enter your text or url link:")

    KB = knowledge_base_NLP(input_text=user_input)
    #st.write(KB.raw_text)

    doc = KB.coref_doc_original
    html = displacy.render(doc, style="ent", jupyter=False)
    st.markdown(html, unsafe_allow_html=True)
    
    st.subheader('extracting relations based on syntactic structure')
    df_relations = KB.pos_relations
    st.write(df_relations)

    st.subheader("extracted attributes")
    df_attributes = KB.extract_attributes
    st.write(df_attributes)


    # Add a button to download the dataframe as an Excel file
    if st.button("Download Relations DataFrame"):
        df_relations.to_excel("relations.xlsx", index=False)
        st.success("Downloaded the Relations DataFrame!")

    if st.button("Download Attributes DataFrame"):
        df_attributes.to_excel("attributes.xlsx", index=False)
        st.success("Downloaded the Attributes DataFrame!")
    


# run
app()
