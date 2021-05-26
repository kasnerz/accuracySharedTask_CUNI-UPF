#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Code used to manipulate json structures of the following type
# Each key should have all of (and only) bubble_feats, form, feats, head, deprel.
# {
    # "change_shape": [
    # {
        # "bubble_feats": "slex=Sentence|parameterChange=yes"
    # },
    # {
        # "form": "change",
        # "feats": {"default": "tense=PAST"},
        # "head": [0],
        # "deprel": "ROOT"
    # },
    # {
        # "form": "[roomParameter]",
        # "feats": {"default": "definiteness=DEF"},
        # "head": [1],
        # "deprel": "A2"
    # }
    # ]
# }

import re
import os, shutil
import json
import glob
import codecs
import numbers
from pathlib import Path


#template_file_path = Path(Path.home(), 'Google Drive', 'TALN', 'Simon', 'Programs', 'Python', 'Generation', 'CONNEXIONs-input-template.json')

json_files_path = "/lnet/work/people/kasner/datasets/rotowire"

file_list = glob.glob(os.path.join(json_files_path, '*.json'))
#file_list_reasoning = glob.glob(os.path.join(json_files_path, '*reasoning.json'))

create_reference_text_file = 'no'

def write_log(message, log_file):
    log_file.write(message+'\n')
    #print(message)

def fillKeysDicoNumber(dico_numbers):
    """ Prepares the dictionary for the compareTeamNumbers function"""
    dico_numbers['winning_team'] = ''
    dico_numbers['losing_team'] = ''
    dico_numbers['score'] = ''
    dico_numbers['score_difference'] = ''

def compareTeamNumbers(dico_numbers, team_data, homeStatCat, visStatCat):
    """ Fills a dictionary with which team has more X (points, rebounds, etc.) than the other team, and the total of Xs and difference between Xs."""
    if int(homeStatCat) < int(visStatCat):    
        dico_numbers['winning_team'] = team_data['vis_city'].team_full_name
        dico_numbers['losing_team'] = team_data['home_city'].team_full_name
        dico_numbers['count_winning_team'] = visStatCat
        dico_numbers['count_losing_team'] = homeStatCat
        dico_numbers['score'] = str(visStatCat)+'-'+str(homeStatCat)
        dico_numbers['score_difference'] = int(visStatCat) - int(homeStatCat)
    else:
        dico_numbers['winning_team'] = team_data['home_city'].team_full_name
        dico_numbers['losing_team'] = team_data['vis_city'].team_full_name
        dico_numbers['count_winning_team'] = homeStatCat
        dico_numbers['count_losing_team'] = visStatCat
        dico_numbers['score'] = str(homeStatCat)+'-'+str(visStatCat)
        dico_numbers['score_difference'] = int(homeStatCat) - int(visStatCat)
    # else:
        # dico_numbers['winning_team'] = '-'
        # dico_numbers['losing_team'] = '-'
        # dico_numbers['count_winning_team'] = visStatCat
        # dico_numbers['count_losing_team'] = homeStatCat
        # dico_numbers['score'] = str(visStatCat)+'-'+str(homeStatCat)
        # dico_numbers['score_difference'] = int(visStatCat) - int(homeStatCat)
    

class PlayerData:
    def __init__(self, json_dict, x, player_id):
        # raw data directly from JSON
        self.PLAYER_NAME = json_dict[x]['box_score']['PLAYER_NAME'][player_id]
        self.FIRST_NAME = json_dict[x]['box_score']['FIRST_NAME'][player_id]
        self.MIN = json_dict[x]['box_score']['MIN'][player_id]
        self.FGM = json_dict[x]['box_score']['FGM'][player_id]
        self.REB = json_dict[x]['box_score']['REB'][player_id]
        self.FG3A = json_dict[x]['box_score']['FG3A'][player_id]
        self.AST = json_dict[x]['box_score']['AST'][player_id]
        self.FG3M = json_dict[x]['box_score']['FG3M'][player_id]
        self.OREB = json_dict[x]['box_score']['OREB'][player_id]
        self.TO = json_dict[x]['box_score']['TO'][player_id]
        self.START_POSITION = 'N/A'
        START_POSITION = json_dict[x]['box_score']['START_POSITION'][player_id]
        if START_POSITION == 'C':
            self.START_POSITION = 'Center'
        elif START_POSITION == 'F':
            self.START_POSITION = 'Forward'
        elif START_POSITION == 'G':
            self.START_POSITION = 'Guard'
        self.PF = json_dict[x]['box_score']['PF'][player_id]
        self.PTS = json_dict[x]['box_score']['PTS'][player_id]
        self.FGA = json_dict[x]['box_score']['FGA'][player_id]
        self.STL = json_dict[x]['box_score']['STL'][player_id]
        self.FTA = json_dict[x]['box_score']['FTA'][player_id]
        self.BLK = json_dict[x]['box_score']['BLK'][player_id]
        self.DREB = json_dict[x]['box_score']['DREB'][player_id]
        self.FTM = json_dict[x]['box_score']['FTM'][player_id]
        self.FT_PCT = json_dict[x]['box_score']['FT_PCT'][player_id]
        self.FG_PCT = json_dict[x]['box_score']['FG_PCT'][player_id]
        self.FG3_PCT = json_dict[x]['box_score']['FG3_PCT'][player_id]
        self.SECOND_NAME = json_dict[x]['box_score']['SECOND_NAME'][player_id]
        self.TEAM_CITY = json_dict[x]['box_score']['TEAM_CITY'][player_id]
        self.DOUBLE_FIGURE = 'no'
        self.DOUBLE_DOUBLE = 'no'
        self.TRIPLE_DOUBLE = 'no'
        self.DOUBLE_DOUBLE_NEAR = 'no'
        self.TRIPLE_DOUBLE_NEAR = 'no'
        self.GOOD_OFF_BENCH = 'no'
        DOUBLE_DIGITS = 0
        PTS_ABOVE_13 = 'no'
        PTS_ABOVE_9 = 'no'
        if self.PTS != 'N/A':
            DOUBLE_DIGITS_ALMOST = 0
            if int(self.PTS) > 13:
                PTS_ABOVE_13 = 'yes'
            if int(self.PTS) > 9:
                PTS_ABOVE_9 = 'yes'
                DOUBLE_DIGITS +=1
            elif int(self.PTS) > 7:
                DOUBLE_DIGITS_ALMOST +=1
        if self.REB != 'N/A':
            if int(self.REB) > 9:
                DOUBLE_DIGITS +=1
            elif int(self.REB) > 7:
                DOUBLE_DIGITS_ALMOST +=1
        if self.AST != 'N/A':
            if int(self.AST) > 9:
                DOUBLE_DIGITS +=1
            elif int(self.AST) > 7:
                DOUBLE_DIGITS_ALMOST +=1
        if self.BLK != 'N/A':
            if int(self.BLK) > 9:
                DOUBLE_DIGITS +=1
            elif int(self.BLK) > 7:
                DOUBLE_DIGITS_ALMOST +=1
        if self.STL != 'N/A':
            if int(self.STL) > 9:
                DOUBLE_DIGITS +=1
            elif int(self.STL) > 7:
                DOUBLE_DIGITS_ALMOST +=1
        if DOUBLE_DIGITS == 1:
            if DOUBLE_DIGITS_ALMOST == 0:
                self.DOUBLE_FIGURE = 'yes'
            elif DOUBLE_DIGITS_ALMOST == 1:
                self.DOUBLE_DOUBLE_NEAR = 'yes'
            elif DOUBLE_DIGITS_ALMOST > 1:
                self.TRIPLE_DOUBLE_NEAR = 'yes'
        elif DOUBLE_DIGITS == 2:
            if DOUBLE_DIGITS_ALMOST == 0:
                self.DOUBLE_DOUBLE = 'yes'
            elif DOUBLE_DIGITS_ALMOST > 0:
                self.TRIPLE_DOUBLE_NEAR = 'yes'
        elif DOUBLE_DIGITS > 2:
            self.TRIPLE_DOUBLE = 'yes'
        if self.START_POSITION == 'N/A' and (PTS_ABOVE_13 == 'yes' or (PTS_ABOVE_9 == 'no' and DOUBLE_DIGITS > 0)):
            self.GOOD_OFF_BENCH = 'yes'
        
class TeamData:
    def __init__(self, json_dict, x, homevis, list_players):
        # raw data directly from JSON
        name = homevis+'_name'
        city = homevis+'_city'
        line = homevis+'_line'
        self.team_name = json_dict[x][name]
        self.team_full_name = json_dict[x][city]+' '+json_dict[x][name]
        self.TEAMFT_PCT = json_dict[x][line]['TEAM-FT_PCT']
        self.TEAMPTS_QTR1 = json_dict[x][line]['TEAM-PTS_QTR1']
        self.TEAMPTS_QTR2 = json_dict[x][line]['TEAM-PTS_QTR2']
        self.TEAMPTS_QTR3 = json_dict[x][line]['TEAM-PTS_QTR3']
        self.TEAMPTS_QTR4 = json_dict[x][line]['TEAM-PTS_QTR4']
        self.TEAMCITY = json_dict[x][line]['TEAM-CITY']
        self.TEAMPTS = json_dict[x][line]['TEAM-PTS']
        self.TEAMAST = json_dict[x][line]['TEAM-AST']
        self.TEAMLOSSES = json_dict[x][line]['TEAM-LOSSES']
        self.TEAMNAME = json_dict[x][line]['TEAM-NAME']
        self.TEAMWINS = json_dict[x][line]['TEAM-WINS']
        self.TEAMREB = json_dict[x][line]['TEAM-REB']
        self.TEAMTOV = json_dict[x][line]['TEAM-TOV']
        self.TEAMFG3_PCT = json_dict[x][line]['TEAM-FG3_PCT']
        self.TEAMFG_PCT = json_dict[x][line]['TEAM-FG_PCT']
        self.players = list_players
        
        #inferred data from combining raw player/team data
        self.TEAMPTS_HALF = str(int(self.TEAMPTS_QTR1) + int(self.TEAMPTS_QTR2))
        self.TEAMPTS_QTR123 = str(int(self.TEAMPTS_QTR1) + int(self.TEAMPTS_QTR2) + int(self.TEAMPTS_QTR3))
        self.TEAMWINLOSS_balance= str(int(self.TEAMWINS) - int(self.TEAMLOSSES))
        self.TEAMGAMES_REMAIN = str(82 - int(self.TEAMWINS) - int(self.TEAMLOSSES))
        self.TEAMFGM = 0
        self.TEAMFGA = 0
        self.TEAMFG3M = 0
        self.TEAMFG3A = 0
        self.TEAMFTA = 0
        self.TEAMFTM = 0
        self.TEAMDREB = 0
        self.TEAMOREB = 0
        self.TEAMPF = 0
        self.TEAMSTL = 0
        self.TEAMPTS_PLAYERS_DOUBLEDIGIT = 0
        self.TEAM_HIGHEST_PTS = 0
        self.TEAM_HIGHEST_REB = 0
        self.TEAM_HIGHEST_AST = 0
        self.TEAM_HIGHEST_BLK = 0
        self.TEAM_HIGHEST_STL = 0
        self.TEAM_HIGHEST_PTS_2nd = 0
        self.TEAM_HIGHEST_REB_2nd = 0
        self.TEAM_HIGHEST_AST_2nd = 0
        self.TEAM_HIGHEST_BLK_2nd = 0
        self.TEAM_HIGHEST_STL_2nd = 0
        for player in self.players:
            if player.FGM != 'N/A':
                self.TEAMFGM += int(player.FGM)
            if player.FGA != 'N/A':
                self.TEAMFGA += int(player.FGA)
            if player.FG3M != 'N/A':
                self.TEAMFG3M += int(player.FG3M)
            if player.FG3A != 'N/A':
                self.TEAMFG3A += int(player.FG3A)
            if player.FTA != 'N/A':
                self.TEAMFTA += int(player.FTA)
            if player.FTM != 'N/A':
                self.TEAMFTM += int(player.FTM)
            if player.DREB != 'N/A':
                self.TEAMDREB += int(player.DREB)
            if player.OREB != 'N/A':
                self.TEAMOREB += int(player.OREB)
            if player.PF != 'N/A':
                self.TEAMPF += int(player.PF)
            if player.STL != 'N/A':
                self.TEAMSTL += int(player.STL)
            if player.PTS != 'N/A':
                if int(player.PTS) > 9:
                    self.TEAMPTS_PLAYERS_DOUBLEDIGIT += 1
                if int(player.PTS) > self.TEAM_HIGHEST_PTS:
                    self.TEAM_HIGHEST_PTS_2nd = self.TEAM_HIGHEST_PTS
                    self.TEAM_HIGHEST_PTS = int(player.PTS)
                elif int(player.PTS) > self.TEAM_HIGHEST_PTS_2nd:
                    self.TEAM_HIGHEST_PTS_2nd = int(player.PTS)
            if player.REB != 'N/A':
                if int(player.REB) > self.TEAM_HIGHEST_REB:
                    self.TEAM_HIGHEST_REB_2nd = self.TEAM_HIGHEST_REB
                    self.TEAM_HIGHEST_REB = int(player.REB)
                elif int(player.REB) > self.TEAM_HIGHEST_REB_2nd:
                    self.TEAM_HIGHEST_REB_2nd = int(player.REB)
            if player.AST != 'N/A':
                if int(player.AST) > self.TEAM_HIGHEST_AST:
                    self.TEAM_HIGHEST_AST_2nd = self.TEAM_HIGHEST_AST
                    self.TEAM_HIGHEST_AST = int(player.AST)
                elif int(player.AST) > self.TEAM_HIGHEST_AST_2nd:
                    self.TEAM_HIGHEST_AST_2nd = int(player.AST)
            if player.BLK != 'N/A':
                if int(player.BLK) > self.TEAM_HIGHEST_BLK:
                    self.TEAM_HIGHEST_BLK_2nd = self.TEAM_HIGHEST_BLK
                    self.TEAM_HIGHEST_BLK = int(player.BLK)
                elif int(player.BLK) > self.TEAM_HIGHEST_BLK_2nd:
                    self.TEAM_HIGHEST_BLK_2nd = int(player.BLK)
            if player.STL != 'N/A':
                if int(player.STL) > self.TEAM_HIGHEST_STL:
                    self.TEAM_HIGHEST_STL_2nd = self.TEAM_HIGHEST_STL
                    self.TEAM_HIGHEST_STL = int(player.STL)
                elif int(player.STL) > self.TEAM_HIGHEST_STL_2nd:
                    self.TEAM_HIGHEST_STL_2nd = int(player.STL)
        self.TEAMFG2M = self.TEAMFGM - self.TEAMFG3M
        self.TEAMFG2A = self.TEAMFGA - self.TEAMFG3A
        self.TEAMFG2_PCT = int(round((self.TEAMFG2M*100)/self.TEAMFG2A, 0))
        
class GameData:
    def __init__(self, home_name, home_city, vis_name, vis_city, day, summary, team_home, team_vis, box_score):
        # raw data directly from JSON
        self.home_name = home_name
        self.home_city = home_city
        self.home_full_name = home_city+' '+home_name
        self.vis_name = vis_name
        self.vis_city = vis_city
        self.vis_full_name = vis_city+' '+vis_name
        self.day = day
        text = ''
        for word in summary:
            text = text + ' ' + word
        self.text = text
        self.teams = {}
        self.teams['home_city'] = team_home
        self.teams['vis_city'] = team_vis
        
        #inferred data from combining raw team data
        #0- General results
        self.result_game = {}
        fillKeysDicoNumber(self.result_game)
        compareTeamNumbers(self.result_game, self.teams, self.teams['home_city'].TEAMPTS, self.teams['vis_city'].TEAMPTS)
        #1- Assists comparison
        self.comparison_AST = {}
        fillKeysDicoNumber(self.comparison_AST)
        compareTeamNumbers(self.comparison_AST, self.teams, self.teams['home_city'].TEAMAST, self.teams['vis_city'].TEAMAST)
        compareTeamNumbers(self.result_game, self.teams, self.teams['home_city'].TEAMPTS, self.teams['vis_city'].TEAMPTS)
        #2- 2 and 3 points comparison
        self.comparison_FGM = {}
        fillKeysDicoNumber(self.comparison_FGM)
        compareTeamNumbers(self.comparison_FGM, self.teams, self.teams['home_city'].TEAMFGM, self.teams['vis_city'].TEAMFGM)
        #3- 2 and 3 points % comparison
        self.comparison_FGPCT = {}
        fillKeysDicoNumber(self.comparison_FGPCT)
        compareTeamNumbers(self.comparison_FGPCT, self.teams, self.teams['home_city'].TEAMFG_PCT, self.teams['vis_city'].TEAMFG_PCT)
        #4- 2 points comparison
        self.comparison_FG2M = {}
        fillKeysDicoNumber(self.comparison_FG2M)
        compareTeamNumbers(self.comparison_FG2M, self.teams, self.teams['home_city'].TEAMFG2M, self.teams['vis_city'].TEAMFG2M)
        #5- 2 points % comparison
        self.comparison_FG2PCT = {}
        fillKeysDicoNumber(self.comparison_FG2PCT)
        compareTeamNumbers(self.comparison_FG2PCT, self.teams, self.teams['home_city'].TEAMFG2_PCT, self.teams['vis_city'].TEAMFG2_PCT)
        #6- 3 points comparison
        self.comparison_FG3M = {}
        fillKeysDicoNumber(self.comparison_FG3M)
        compareTeamNumbers(self.comparison_FG3M, self.teams, self.teams['home_city'].TEAMFG3M, self.teams['vis_city'].TEAMFG3M)
        #7- 3 points % comparison
        self.comparison_FG3PCT = {}
        fillKeysDicoNumber(self.comparison_FG3PCT)
        compareTeamNumbers(self.comparison_FG3PCT, self.teams, self.teams['home_city'].TEAMFG3_PCT, self.teams['vis_city'].TEAMFG3_PCT)
        #8- Free throws made comparison
        self.comparison_FTM = {}
        fillKeysDicoNumber(self.comparison_FTM)
        compareTeamNumbers(self.comparison_FTM, self.teams, self.teams['home_city'].TEAMFTM, self.teams['vis_city'].TEAMFTM)
        #9- Free throws attempted comparison
        self.comparison_FTA = {}
        fillKeysDicoNumber(self.comparison_FTA)
        compareTeamNumbers(self.comparison_FTA, self.teams, self.teams['home_city'].TEAMFTA, self.teams['vis_city'].TEAMFTA)
        #10- Free throws % comparison
        self.comparison_FTPCT = {}
        fillKeysDicoNumber(self.comparison_FTPCT)
        compareTeamNumbers(self.comparison_FTPCT, self.teams, self.teams['home_city'].TEAMFT_PCT, self.teams['vis_city'].TEAMFT_PCT)
        #11- Rebounds comparison
        self.comparison_REB = {}
        fillKeysDicoNumber(self.comparison_REB)
        compareTeamNumbers(self.comparison_REB, self.teams, self.teams['home_city'].TEAMREB, self.teams['vis_city'].TEAMREB)
        #12- Rebounds comparison
        self.comparison_DREB = {}
        fillKeysDicoNumber(self.comparison_DREB)
        compareTeamNumbers(self.comparison_DREB, self.teams, self.teams['home_city'].TEAMDREB, self.teams['vis_city'].TEAMDREB)
        #13- Rebounds comparison
        self.comparison_OREB = {}
        fillKeysDicoNumber(self.comparison_OREB)
        compareTeamNumbers(self.comparison_OREB, self.teams, self.teams['home_city'].TEAMOREB, self.teams['vis_city'].TEAMOREB)
        #14- Fouls and turnovers comparison
        self.comparison_PF = {}
        fillKeysDicoNumber(self.comparison_PF)
        compareTeamNumbers(self.comparison_PF, self.teams, self.teams['home_city'].TEAMPF, self.teams['vis_city'].TEAMPF)
        self.comparison_TOV = {}
        fillKeysDicoNumber(self.comparison_TOV)
        compareTeamNumbers(self.comparison_TOV, self.teams, self.teams['home_city'].TEAMTOV, self.teams['vis_city'].TEAMTOV)
        #15- Steals comparison
        self.comparison_STL = {}
        fillKeysDicoNumber(self.comparison_STL)
        compareTeamNumbers(self.comparison_STL, self.teams, self.teams['home_city'].TEAMSTL, self.teams['vis_city'].TEAMSTL)
        #16- Results Q1
        self.result_Q1 = {}
        fillKeysDicoNumber(self.result_Q1)
        compareTeamNumbers(self.result_Q1, self.teams, self.teams['home_city'].TEAMPTS_QTR1, self.teams['vis_city'].TEAMPTS_QTR1)
        #17- Results Q2
        self.result_Q2 = {}
        fillKeysDicoNumber(self.result_Q2)
        compareTeamNumbers(self.result_Q2, self.teams, self.teams['home_city'].TEAMPTS_QTR2, self.teams['vis_city'].TEAMPTS_QTR2)
        #18- Results Q3
        self.result_Q3 = {}
        fillKeysDicoNumber(self.result_Q3)
        compareTeamNumbers(self.result_Q3, self.teams, self.teams['home_city'].TEAMPTS_QTR3, self.teams['vis_city'].TEAMPTS_QTR3)
        #19- Results Q4
        self.result_Q4 = {}
        fillKeysDicoNumber(self.result_Q4)
        compareTeamNumbers(self.result_Q4, self.teams, self.teams['home_city'].TEAMPTS_QTR4, self.teams['vis_city'].TEAMPTS_QTR4)
        #20- Results HalfTime
        self.result_halftime = {}
        fillKeysDicoNumber(self.result_halftime)
        compareTeamNumbers(self.result_halftime, self.teams, self.teams['home_city'].TEAMPTS_HALF, self.teams['vis_city'].TEAMPTS_HALF)
        #21- Favorite
        self.favorite_team = ''
        if int(self.teams['home_city'].TEAMWINLOSS_balance) > int(self.teams['vis_city'].TEAMWINLOSS_balance):
            self.favorite_team = self.home_city
        elif int(self.teams['home_city'].TEAMWINLOSS_balance) < int(self.teams['vis_city'].TEAMWINLOSS_balance):
            self.favorite_team = self.vis_city
        else:
            self.favorite_team = 'No one'
        #21- Come back
        self.comeback_team = ''
        if (int(self.teams['home_city'].TEAMPTS_QTR123) - int(self.teams['vis_city'].TEAMPTS_QTR123)) > 9:
            if self.result_game['winning_team'] == self.teams['vis_city'].team_full_name:
                self.comeback_team = self.vis_city
            else:
                pass
        elif (int(self.teams['vis_city'].TEAMPTS_QTR123) - int(self.teams['home_city'].TEAMPTS_QTR123)) > 9:
            if self.result_game['winning_team'] == self.teams['home_city'].team_full_name:
                self.comeback_team = self.home_city
            else:
                pass
        #22- Highest stats
        self.HIGHEST_PTS = 0
        self.HIGHEST_REB = 0
        self.HIGHEST_AST = 0
        self.HIGHEST_BLK = 0
        self.HIGHEST_STL = 0
        self.HIGHEST_PTS_2nd = 0
        self.HIGHEST_REB_2nd = 0
        self.HIGHEST_AST_2nd = 0
        self.HIGHEST_BLK_2nd = 0
        self.HIGHEST_STL_2nd = 0
        for player_id in box_score['FIRST_NAME']:
            if box_score['PTS'][player_id] != 'N/A':
                count = int(box_score['PTS'][player_id])
                if count > self.HIGHEST_PTS:
                    self.HIGHEST_PTS_2nd = self.HIGHEST_PTS
                    self.HIGHEST_PTS = count
                elif count > self.HIGHEST_PTS_2nd:
                    self.HIGHEST_PTS_2nd = count
            if box_score['REB'][player_id] != 'N/A':
                count = int(box_score['REB'][player_id])
                if count > self.HIGHEST_REB:
                    self.HIGHEST_REB_2nd = self.HIGHEST_REB
                    self.HIGHEST_REB = count
                elif count > self.HIGHEST_REB_2nd:
                    self.HIGHEST_REB_2nd = count
            if box_score['AST'][player_id] != 'N/A':
                count = int(box_score['AST'][player_id])
                if count > self.HIGHEST_AST:
                    self.HIGHEST_AST_2nd = self.HIGHEST_AST
                    self.HIGHEST_AST = count
                elif count > self.HIGHEST_AST_2nd:
                    self.HIGHEST_AST_2nd = count
            if box_score['BLK'][player_id] != 'N/A':
                count = int(box_score['BLK'][player_id])
                if count > self.HIGHEST_BLK:
                    self.HIGHEST_BLK_2nd = self.HIGHEST_BLK
                    self.HIGHEST_BLK = count
                elif count > self.HIGHEST_BLK_2nd:
                    self.HIGHEST_BLK_2nd = count
            if box_score['STL'][player_id] != 'N/A':
                count = int(box_score['STL'][player_id])
                if count > self.HIGHEST_STL:
                    self.HIGHEST_STL_2nd = self.HIGHEST_STL
                    self.HIGHEST_STL = count
                elif count > self.HIGHEST_STL_2nd:
                    self.HIGHEST_STL_2nd = count
            
try:
    shutil.rmtree('out')
except Exception as e:
    pass
os.makedirs('out')
    
# Read profiling files and extract relevant data to generate from. The input featires and values are stored as objects in a list.
#print(len(file_list), 'profiling input files have been found.')
#print(len(file_list_reasoning), 'reasoning input files have been found.')

def create_json_template (file_list):
    for json_file_path in file_list:

        filename = json_file_path.split('/')[-1].split('.')[0]
        filename_out = 'log_'+filename+'.txt'
        print('Writing file '+filename_out)
        log_file = codecs.open(out+"/"+filename_out, 'w', 'utf-8')
        list_games = []
        write_log('### File ### '+json_file_path, log_file)
        
        with open(json_file_path, encoding='utf-8') as f:
            json_dict = json.load(f)
            x = 0
            while x < len(json_dict):
                list_home_players = []
                list_vis_players = []
                
                # It happens that two teams have the same name (Los Angeles), it's not possible to distinguish the players of each team in this case since the team name is used in Rotowire
                if json_dict[x]['home_city'] == json_dict[x]['vis_city']:
                    write_log('Input #'+str(x)+' not processed: '+json_dict[x]['day']+' - '+json_dict[x]['home_city']+'/'+json_dict[x]['vis_city'], log_file)
                else:
                    # fill in raw player data
                    for player_id in json_dict[x]['box_score']['FIRST_NAME']:
                        player = PlayerData(json_dict, x, player_id)
                        if json_dict[x]['box_score']['TEAM_CITY'][player_id] == json_dict[x]['home_city']:
                            # team_home.players.append(player)
                            list_home_players.append(player)
                        elif json_dict[x]['box_score']['TEAM_CITY'][player_id] == json_dict[x]['vis_city']:
                            # team_vis.players.append(player)
                            list_vis_players.append(player)
                    
                    # fill in raw team data
                    team_home = TeamData(json_dict, x, 'home', list_home_players)
                    team_vis = TeamData(json_dict, x, 'vis', list_vis_players)
                    
                    # fill in raw game data
                    game = GameData(json_dict[x]['home_name'], json_dict[x]['home_city'], json_dict[x]['vis_name'], json_dict[x]['vis_city'], json_dict[x]['day'], json_dict[x]['summary'], team_home, team_vis, json_dict[x]['box_score'])
                    
                    list_games.append(game)

                x += 1
        write_log(str(len(list_games))+' valid inputs found out of '+str(len(json_dict)), log_file)
        
        # Print output data to see how it looks
        count_games = 0
        for game in list_games:
            write_log('\n=== Game #'+str(count_games)+' ===', log_file)
            write_log('--- Game Data ---', log_file)
            write_log(game.home_full_name+' hosted '+game.vis_full_name+' on '+game.day+'.', log_file)
            write_log(game.result_game['winning_team']+' beat '+game.result_game['losing_team']+' by '+str(game.result_game['score_difference'])+' points ('+game.result_game['score']+').', log_file)
            if game.comparison_AST['score_difference'] == 0:
                write_log(game.comparison_AST['winning_team']+' and '+str(game.comparison_AST['losing_team'])+' had the same number of assists ('+str(game.comparison_AST['count_winning_team'])+').', log_file)
            else:
                write_log(game.comparison_AST['winning_team']+' got '+str(game.comparison_AST['score_difference'])+' more assists than '+game.comparison_AST['losing_team']+' ('+game.comparison_AST['score']+').', log_file)
            if game.comparison_FGM['score_difference'] == 0:
                write_log(game.comparison_FGM['winning_team']+' and '+str(game.comparison_FGM['losing_team'])+' scored the same amount of baskets ('+str(game.comparison_FGM['count_winning_team'])+').', log_file)
            else:
                write_log(game.comparison_FGM['winning_team']+' scored '+str(game.comparison_FGM['score_difference'])+' more baskets than '+game.comparison_FGM['losing_team']+' ('+game.comparison_FGM['score']+').', log_file)
            if game.comparison_FGPCT['score_difference'] == 0:
                write_log(game.comparison_FGPCT['winning_team']+' and '+str(game.comparison_FGPCT['losing_team'])+' achieved the same shooting percentage ('+str(game.comparison_FGPCT['count_winning_team'])+'%).', log_file)
            else:
                write_log(game.comparison_FGPCT['winning_team']+' shot better than '+game.comparison_FGPCT['losing_team']+' ('+str(game.comparison_FGPCT['count_winning_team'])+'% to '+str(game.comparison_FGPCT['count_losing_team'])+'%).', log_file)
            if game.comparison_FG2M['score_difference'] == 0:
                write_log(game.comparison_FG2M['winning_team']+' and '+str(game.comparison_FG2M['losing_team'])+' scored the same amount of baskets in the paint ('+str(game.comparison_FG2M['count_winning_team'])+').', log_file)
            else:
                write_log(game.comparison_FG2M['winning_team']+' scored '+str(game.comparison_FG2M['score_difference'])+' more baskets in the paint than '+game.comparison_FG2M['losing_team']+' ('+game.comparison_FG2M['score']+').', log_file)
            if game.comparison_FG2PCT['score_difference'] == 0:
                write_log(game.comparison_FG2PCT['winning_team']+' and '+str(game.comparison_FG2PCT['losing_team'])+' achieved the same shooting percentage in the paint ('+str(game.comparison_FG2PCT['count_winning_team'])+'%).', log_file)
            else:
                write_log(game.comparison_FG2PCT['winning_team']+' shot better than '+game.comparison_FG2PCT['losing_team']+' in the paint ('+str(game.comparison_FG2PCT['count_winning_team'])+'% to '+str(game.comparison_FG2PCT['count_losing_team'])+'%).', log_file)
            if game.comparison_FG3M['score_difference'] == 0:
                write_log(game.comparison_FG3M['winning_team']+' and '+str(game.comparison_FG3M['losing_team'])+' scored the same amount of baskets behind the arc ('+str(game.comparison_FG3M['count_winning_team'])+').', log_file)
            else:
                write_log(game.comparison_FG3M['winning_team']+' scored '+str(game.comparison_FG3M['score_difference'])+' more baskets behind the arc than '+game.comparison_FG3M['losing_team']+' ('+game.comparison_FG3M['score']+').', log_file)
            if game.comparison_FG3PCT['score_difference'] == 0:
                write_log(game.comparison_FG3PCT['winning_team']+' and '+str(game.comparison_FG3PCT['losing_team'])+' achieved the same shooting percentage behind the arc ('+str(game.comparison_FG3PCT['count_winning_team'])+'%).', log_file)
            else:
                write_log(game.comparison_FG3PCT['winning_team']+' shot better than '+game.comparison_FG3PCT['losing_team']+' behind the arc ('+str(game.comparison_FG3PCT['count_winning_team'])+'% to '+str(game.comparison_FG3PCT['count_losing_team'])+'%).', log_file)
            if game.comparison_FTM['score_difference'] == 0:
                write_log(game.comparison_FTM['winning_team']+' and '+str(game.comparison_FTM['losing_team'])+' scored the same amount of free throws ('+str(game.comparison_FTM['count_winning_team'])+').', log_file)
            else:
                write_log(game.comparison_FTM['winning_team']+' drained '+str(game.comparison_FTM['score_difference'])+' more free throws than '+game.comparison_FTM['losing_team']+' ('+game.comparison_FTM['score']+').', log_file)
            if game.comparison_FTA['score_difference'] == 0:
                write_log(game.comparison_FTA['winning_team']+' and '+str(game.comparison_FTA['losing_team'])+' gave up the same amount of free throws ('+str(game.comparison_FTA['count_winning_team'])+').', log_file)
            else:
                write_log(game.comparison_FTA['losing_team']+' gave up '+str(game.comparison_FTA['score_difference'])+' more free throws than '+game.comparison_FTA['winning_team']+' ('+game.comparison_FTA['score']+').', log_file)
            if game.comparison_FTPCT['score_difference'] == 0:
                write_log(game.comparison_FTPCT['winning_team']+' and '+str(game.comparison_FTPCT['losing_team'])+' achieved the same shooting percentage from the line ('+str(game.comparison_FTPCT['count_winning_team'])+'%).', log_file)
            else:
                write_log(game.comparison_FTPCT['winning_team']+' shot better than '+game.comparison_FTPCT['losing_team']+' from the line ('+str(game.comparison_FTPCT['count_winning_team'])+'% to '+str(game.comparison_FTPCT['count_losing_team'])+'%).', log_file)
            if game.comparison_REB['score_difference'] == 0:
                write_log(game.comparison_REB['winning_team']+' and '+str(game.comparison_REB['losing_team'])+' had the same number of rebounds ('+str(game.comparison_REB['count_winning_team'])+').', log_file)
            else:
                write_log(game.comparison_REB['winning_team']+' got '+str(game.comparison_REB['score_difference'])+' more rebounds than '+game.comparison_REB['losing_team']+' ('+game.comparison_REB['score']+').', log_file)
            if game.comparison_DREB['score_difference'] == 0:
                write_log(game.comparison_DREB['winning_team']+' and '+str(game.comparison_DREB['losing_team'])+' had the same number of defensive rebounds ('+str(game.comparison_DREB['count_winning_team'])+').', log_file)
            else:
                write_log(game.comparison_DREB['winning_team']+' got '+str(game.comparison_DREB['score_difference'])+' more defensive rebounds than '+game.comparison_DREB['losing_team']+' ('+game.comparison_DREB['score']+').', log_file)
            if game.comparison_OREB['score_difference'] == 0:
                write_log(game.comparison_OREB['winning_team']+' and '+str(game.comparison_OREB['losing_team'])+' had the same number of offensive rebounds ('+str(game.comparison_OREB['count_winning_team'])+').', log_file)
            else:
                write_log(game.comparison_OREB['winning_team']+' got '+str(game.comparison_OREB['score_difference'])+' more offensive rebounds than '+game.comparison_OREB['losing_team']+' ('+game.comparison_OREB['score']+').', log_file)
            if game.comparison_PF['score_difference'] == 0:
                write_log(game.comparison_PF['winning_team']+' and '+str(game.comparison_PF['losing_team'])+' committed the same number of fouls ('+str(game.comparison_PF['count_winning_team'])+').', log_file)
            else:
                write_log(game.comparison_PF['winning_team']+' committed '+str(game.comparison_PF['score_difference'])+' more fouls than '+game.comparison_PF['losing_team']+' ('+game.comparison_PF['score']+').', log_file)
            if game.comparison_TOV['score_difference'] == 0:
                write_log(game.comparison_TOV['winning_team']+' and '+str(game.comparison_TOV['losing_team'])+' committed the same number of turnovers ('+str(game.comparison_TOV['count_winning_team'])+').', log_file)
            else:
                write_log(game.comparison_TOV['winning_team']+' committed '+str(game.comparison_TOV['score_difference'])+' more turnovers than '+game.comparison_TOV['losing_team']+' ('+game.comparison_TOV['score']+').', log_file)
            if game.comparison_STL['score_difference'] == 0:
                write_log(game.comparison_STL['winning_team']+' and '+str(game.comparison_STL['losing_team'])+' had the same number of steals ('+str(game.comparison_STL['count_winning_team'])+').', log_file)
            else:
                write_log(game.comparison_STL['winning_team']+' had '+str(game.comparison_STL['score_difference'])+' more steals than '+game.comparison_STL['losing_team']+' ('+game.comparison_STL['score']+').', log_file)
            if game.result_Q1['score_difference'] == 0:
                write_log(game.result_Q1['winning_team']+' and '+str(game.result_Q1['losing_team'])+' drew the first quarter at '+game.result_Q1['score']+'.', log_file)
            else:
                write_log(game.result_Q1['winning_team']+' won the first quarter by '+str(game.result_Q1['score_difference'])+' points ('+game.result_Q1['score']+').', log_file)
            if game.result_Q2['score_difference'] == 0:
                write_log(game.result_Q2['winning_team']+' and '+str(game.result_Q2['losing_team'])+' drew the second quarter at '+game.result_Q2['score']+'.', log_file)
            else:
                write_log(game.result_Q2['winning_team']+' won the second quarter by '+str(game.result_Q2['score_difference'])+' points ('+game.result_Q2['score']+').', log_file)
            if game.result_Q3['score_difference'] == 0:
                write_log(game.result_Q3['winning_team']+' and '+str(game.result_Q3['losing_team'])+' drew the third quarter at '+game.result_Q3['score']+'.', log_file)
            else:
                write_log(game.result_Q3['winning_team']+' won the third quarter by '+str(game.result_Q3['score_difference'])+' points ('+game.result_Q3['score']+').', log_file)
            if game.result_Q4['score_difference'] == 0:
                write_log(game.result_Q4['winning_team']+' and '+str(game.result_Q4['losing_team'])+' drew the fourth quarter at '+game.result_Q4['score']+'.', log_file)
            else:
                write_log(game.result_Q4['winning_team']+' won the fourth quarter by '+str(game.result_Q4['score_difference'])+' points ('+game.result_Q4['score']+').', log_file)
            if game.result_halftime['score_difference'] == 0:
                write_log(game.result_halftime['winning_team']+' and '+str(game.result_halftime['losing_team'])+' drew the first half at '+game.result_halftime['score']+'.', log_file)
            else:
                write_log(game.result_halftime['winning_team']+' won the first half by '+str(game.result_halftime['score_difference'])+' points ('+game.result_halftime['score']+').', log_file)
            write_log(game.favorite_team+' was the favorite in this game.', log_file)
            
            write_log('--- Team Data ---', log_file)
            if game.comeback_team:
                write_log(game.comeback_team+' embarked in a come back in the fourth quarter.', log_file)
            write_log(game.home_full_name+' had '+str(game.teams['home_city'].TEAMPTS_PLAYERS_DOUBLEDIGIT)+' players score in double digits.', log_file)
            write_log(game.vis_full_name+' had '+str(game.teams['vis_city'].TEAMPTS_PLAYERS_DOUBLEDIGIT)+' players score in double digits.', log_file)
            write_log(game.home_full_name+' have '+str(game.teams['home_city'].TEAMGAMES_REMAIN)+' games remaining.', log_file)
            write_log(game.vis_full_name+' have '+str(game.teams['vis_city'].TEAMGAMES_REMAIN)+' games remaining.', log_file)
            # print(game.home_city+' '+game.home_full_name)
            # for player in game.teams['home_city'].players:
                # print('   '+player.PLAYER_NAME)
            # print(game.vis_city+' '+game.vis_full_name)
            # for player in game.teams['vis_city'].players:
                # print('   '+player.PLAYER_NAME)
            write_log(game.home_full_name+' had '+str(game.teams['home_city'].TEAMAST)+' assists.', log_file)
            write_log(game.vis_full_name+' had '+str(game.teams['vis_city'].TEAMAST)+' assists.', log_file)
            write_log(game.home_full_name+' are from '+str(game.teams['home_city'].TEAMCITY)+'.', log_file)
            write_log(game.vis_full_name+' are from '+str(game.teams['vis_city'].TEAMCITY)+'.', log_file)
            write_log(game.home_full_name+' collected '+str(game.teams['home_city'].TEAMDREB)+' defensive rebounds.', log_file)
            write_log(game.vis_full_name+' collected '+str(game.teams['vis_city'].TEAMDREB)+' defensive rebounds.', log_file)
            write_log(game.home_full_name+' shot '+str(game.teams['home_city'].TEAMFG_PCT)+'% from the field.', log_file)
            if int(game.teams['home_city'].TEAMFG_PCT) > 50:
                write_log(game.home_full_name+' were efficient with the shooting.', log_file)
            elif int(game.teams['home_city'].TEAMFG_PCT) < 40:
                write_log(game.home_full_name+' were bad at shooting.', log_file)
            write_log(game.vis_full_name+' shot '+str(game.teams['vis_city'].TEAMFG_PCT)+'% from the field.', log_file)
            if int(game.teams['vis_city'].TEAMFG_PCT) > 50:
                write_log(game.vis_full_name+' were efficient with the shooting.', log_file)
            elif int(game.teams['vis_city'].TEAMFG_PCT) < 40:
                write_log(game.vis_full_name+' were bad at shooting.', log_file)
            write_log(game.home_full_name+' shot '+str(game.teams['home_city'].TEAMFG2_PCT)+'% from the paint.', log_file)
            write_log(game.vis_full_name+' shot '+str(game.teams['vis_city'].TEAMFG2_PCT)+'% from the paint.', log_file)
            write_log(game.home_full_name+' attempted '+str(game.teams['home_city'].TEAMFG2A)+' shots from the paint.', log_file)
            write_log(game.vis_full_name+' attempted '+str(game.teams['vis_city'].TEAMFG2A)+' shots from the paint.', log_file)
            write_log(game.home_full_name+' scored '+str(game.teams['home_city'].TEAMFG2M)+' baskets from the paint.', log_file)
            write_log(game.vis_full_name+' scored '+str(game.teams['vis_city'].TEAMFG2M)+' baskets from the paint.', log_file)
            write_log(game.home_full_name+' shot '+str(game.teams['home_city'].TEAMFG3_PCT)+' % from beyond the arc.', log_file)
            write_log(game.vis_full_name+' shot '+str(game.teams['vis_city'].TEAMFG3_PCT)+' % from beyond the arc.', log_file)
            write_log(game.home_full_name+' attempted '+str(game.teams['home_city'].TEAMFG3A)+' shots from beyond the arc.', log_file)
            write_log(game.vis_full_name+' attempted '+str(game.teams['vis_city'].TEAMFG3A)+' shots from beyond the arc.', log_file)
            write_log(game.home_full_name+' scored '+str(game.teams['home_city'].TEAMFG3M)+' baskets from beyond the arc.', log_file)
            write_log(game.vis_full_name+' scored '+str(game.teams['vis_city'].TEAMFG3M)+' baskets from beyond the arc.', log_file)
            write_log(game.home_full_name+' attempted '+str(game.teams['home_city'].TEAMFGA)+' shots from the field.', log_file)
            write_log(game.vis_full_name+' attempted '+str(game.teams['vis_city'].TEAMFGA)+' shots from the field.', log_file)
            write_log(game.home_full_name+' scored '+str(game.teams['home_city'].TEAMFGM)+' baskets from the field.', log_file)
            write_log(game.vis_full_name+' scored '+str(game.teams['vis_city'].TEAMFGM)+' baskets from the field.', log_file)
            write_log(game.home_full_name+' shot '+str(game.teams['home_city'].TEAMFT_PCT)+' % from the line.', log_file)
            write_log(game.vis_full_name+' shot '+str(game.teams['vis_city'].TEAMFT_PCT)+' % from the line.', log_file)
            write_log(game.home_full_name+' attempted '+str(game.teams['home_city'].TEAMFTA)+' free throws.', log_file)
            write_log(game.vis_full_name+' attempted '+str(game.teams['vis_city'].TEAMFTA)+' free throws.', log_file)
            write_log(game.home_full_name+' scored '+str(game.teams['home_city'].TEAMFTM)+' free throws.', log_file)
            write_log(game.vis_full_name+' scored '+str(game.teams['vis_city'].TEAMFTM)+' free throws.', log_file)
            write_log(game.home_full_name+' lost '+str(game.teams['home_city'].TEAMLOSSES)+' games.', log_file)
            write_log(game.vis_full_name+' lost '+str(game.teams['vis_city'].TEAMLOSSES)+' games.', log_file)
            write_log(game.home_full_name+' collected '+str(game.teams['home_city'].TEAMOREB)+' offensive rebounds.', log_file)
            write_log(game.vis_full_name+' collected '+str(game.teams['vis_city'].TEAMOREB)+' offensive rebounds.', log_file)
            write_log(game.home_full_name+' committed '+str(game.teams['home_city'].TEAMPF)+' fouls.', log_file)
            write_log(game.vis_full_name+' committed '+str(game.teams['vis_city'].TEAMPF)+' fouls.', log_file)
            write_log(game.home_full_name+' scored '+str(game.teams['home_city'].TEAMPTS)+' points.', log_file)
            write_log(game.vis_full_name+' scored '+str(game.teams['vis_city'].TEAMPTS)+' points.', log_file)
            write_log(game.home_full_name+' scored '+str(game.teams['home_city'].TEAMPTS_HALF)+' points in the first half.', log_file)
            write_log(game.vis_full_name+' scored '+str(game.teams['vis_city'].TEAMPTS_HALF)+' points in the first half.', log_file)
            write_log(game.home_full_name+' scored '+str(game.teams['home_city'].TEAMPTS_QTR1)+' points in the first quarter.', log_file)
            write_log(game.vis_full_name+' scored '+str(game.teams['vis_city'].TEAMPTS_QTR1)+' points in the first quarter.', log_file)
            write_log(game.home_full_name+' scored '+str(game.teams['home_city'].TEAMPTS_QTR123)+' points in the first three quarters.', log_file)
            write_log(game.vis_full_name+' scored '+str(game.teams['vis_city'].TEAMPTS_QTR123)+' points in the first three quarters.', log_file)
            write_log(game.home_full_name+' scored '+str(game.teams['home_city'].TEAMPTS_QTR2)+' points in the second quarter.', log_file)
            write_log(game.vis_full_name+' scored '+str(game.teams['vis_city'].TEAMPTS_QTR2)+' points in the second quarter.', log_file)
            write_log(game.home_full_name+' scored '+str(game.teams['home_city'].TEAMPTS_QTR3)+' points in the third quarter.', log_file)
            write_log(game.vis_full_name+' scored '+str(game.teams['vis_city'].TEAMPTS_QTR3)+' points in the third quarter.', log_file)
            write_log(game.home_full_name+' scored '+str(game.teams['home_city'].TEAMPTS_QTR4)+' points in the fourth quarter.', log_file)
            write_log(game.vis_full_name+' scored '+str(game.teams['vis_city'].TEAMPTS_QTR4)+' points in the fourth quarter.', log_file)
            write_log(game.home_full_name+' collected '+str(game.teams['home_city'].TEAMREB)+' rebounds in total.', log_file)
            write_log(game.vis_full_name+' collected '+str(game.teams['vis_city'].TEAMREB)+' rebounds in total.', log_file)
            write_log(game.home_full_name+' had '+str(game.teams['home_city'].TEAMSTL)+' steals.', log_file)
            write_log(game.vis_full_name+' had '+str(game.teams['vis_city'].TEAMSTL)+' steals.', log_file)
            write_log(game.home_full_name+' committed '+str(game.teams['home_city'].TEAMTOV)+' turnovers.', log_file)
            write_log(game.vis_full_name+' committed '+str(game.teams['vis_city'].TEAMTOV)+' turnovers.', log_file)
            write_log(game.home_full_name+' have a win-loss balance of '+str(game.teams['home_city'].TEAMWINLOSS_balance)+'.', log_file)
            write_log(game.vis_full_name+' have a win-loss balance of '+str(game.teams['vis_city'].TEAMWINLOSS_balance)+'.', log_file)
            write_log(game.home_full_name+' won '+str(game.teams['home_city'].TEAMWINS)+' games.', log_file)
            write_log(game.vis_full_name+' won '+str(game.teams['vis_city'].TEAMWINS)+' games.', log_file)
            write_log(game.home_full_name+' lost '+str(game.teams['home_city'].TEAMLOSSES)+' games.', log_file)
            write_log(game.vis_full_name+' lost '+str(game.teams['vis_city'].TEAMLOSSES)+' games.', log_file)
            
            
            write_log('--- Player Data ---', log_file)
            list_teams = ['home_city', 'vis_city']
            for team in list_teams:
                for player in game.teams[team].players:
                    if player.MIN == 'N/A':
                        write_log(player.PLAYER_NAME+' did not play.', log_file)
                    else:
                        if player.MIN == 0:
                            write_log(player.PLAYER_NAME+' did not play.', log_file)
                        else:
                            if player.AST != 'N/A':
                                write_log(player.PLAYER_NAME+' provided '+str(player.AST)+' assists.', log_file)
                                if int(player.AST) > 0:
                                    if int(player.AST) == int(game.teams[team].TEAM_HIGHEST_AST):
                                        write_log(player.PLAYER_NAME+' was the player with most assists of '+str(player.TEAM_CITY), log_file)
                                    elif int(player.AST) == int(game.teams[team].TEAM_HIGHEST_AST_2nd):
                                        write_log(player.PLAYER_NAME+' was the second player with most assists of '+str(player.TEAM_CITY), log_file)
                                    if int(player.AST) == int(game.HIGHEST_AST):
                                        write_log(player.PLAYER_NAME+' was the player with most assists in the game.', log_file)
                                    elif int(player.AST) == int(game.HIGHEST_AST_2nd):
                                        write_log(player.PLAYER_NAME+' was the second player with most assists in the game.', log_file)
                            if player.BLK != 'N/A':
                                write_log(player.PLAYER_NAME+' provided '+str(player.BLK)+' blocks.', log_file)
                                if int(player.BLK) > 0:
                                    if int(player.BLK) == int(game.teams[team].TEAM_HIGHEST_BLK):
                                        write_log(player.PLAYER_NAME+' was the player with most blocks of '+str(player.TEAM_CITY), log_file)
                                    # elif int(player.BLK) == int(game.teams[team].TEAM_HIGHEST_BLK_2nd):
                                        # write_log(player.PLAYER_NAME+' was the second player with most blocks of '+str(player.TEAM_CITY), log_file)
                                    if int(player.BLK) == int(game.HIGHEST_BLK):
                                        write_log(player.PLAYER_NAME+' was the player with most blocks in the game.', log_file)
                                    # elif int(player.BLK) == int(game.HIGHEST_BLK_2nd):
                                        # write_log(player.PLAYER_NAME+' was the second player with most blocks in the game.', log_file)
                            if player.DREB != 'N/A':
                                write_log(player.PLAYER_NAME+' provided '+str(player.DREB)+' defensive rebounds.', log_file)
                            if player.FG_PCT != 'N/A':
                                if player.FGA != 'N/A':
                                    if int(player.FGA) > 0:
                                        write_log(player.PLAYER_NAME+' shot '+str(player.FG_PCT)+'% from the field.', log_file)
                            if player.FG3_PCT != 'N/A':
                                if player.FG3A != 'N/A':
                                    if int(player.FG3A) > 0:
                                        write_log(player.PLAYER_NAME+' shot '+str(player.FG3_PCT)+'% from beyond the arc.', log_file)
                            if player.FG3A != 'N/A':
                                write_log(player.PLAYER_NAME+' attempted '+str(player.FG3A)+' shots beyond the arc.', log_file)
                            if player.FG3M != 'N/A':
                                write_log(player.PLAYER_NAME+' scored '+str(player.FG3M)+' baskets from beyond the arc.', log_file)
                            if player.FGA != 'N/A':
                                write_log(player.PLAYER_NAME+' attempted '+str(player.FGA)+' shots from the field.', log_file)
                            if player.FGM != 'N/A':
                                write_log(player.PLAYER_NAME+' scored '+str(player.FGM)+' baskets from the field.', log_file)
                            if player.FT_PCT != 'N/A':
                                if player.FTA != 'N/A':
                                    if int(player.FTA) > 0:
                                        write_log(player.PLAYER_NAME+' shot '+str(player.FT_PCT)+'% from the line.', log_file)
                            if player.FTA != 'N/A':
                                write_log(player.PLAYER_NAME+' attempted '+str(player.FTA)+' free throws.', log_file)
                            if player.FTM != 'N/A':
                                write_log(player.PLAYER_NAME+' scored '+str(player.FTM)+' free throws.', log_file)
                            if player.MIN != 'N/A':
                                write_log(player.PLAYER_NAME+' played '+str(player.MIN)+' minutes.', log_file)
                            if player.OREB != 'N/A':
                                write_log(player.PLAYER_NAME+' provided '+str(player.OREB)+' offensive rebounds.', log_file)
                            if player.PF != 'N/A':
                                write_log(player.PLAYER_NAME+' committed '+str(player.PF)+' fouls.', log_file)
                            if player.PTS != 'N/A':
                                write_log(player.PLAYER_NAME+' scored '+str(player.PTS)+' points.', log_file)
                                if int(player.PTS) == int(game.teams[team].TEAM_HIGHEST_PTS):
                                    write_log(player.PLAYER_NAME+' was the best scorer of '+str(player.TEAM_CITY), log_file)
                                elif int(player.PTS) == int(game.teams[team].TEAM_HIGHEST_PTS_2nd):
                                    write_log(player.PLAYER_NAME+' was the second best scorer of '+str(player.TEAM_CITY), log_file)
                                if int(player.PTS) == int(game.HIGHEST_PTS):
                                    write_log(player.PLAYER_NAME+' was the best scorer in the game.', log_file)
                                elif int(player.PTS) == int(game.HIGHEST_PTS_2nd):
                                    write_log(player.PLAYER_NAME+' was the second best scorer in the game.', log_file)
                            if player.REB != 'N/A':
                                write_log(player.PLAYER_NAME+' provided  '+str(player.REB)+' rebounds.', log_file)
                                if int(player.REB) > 0:
                                    if int(player.REB) == int(game.teams[team].TEAM_HIGHEST_REB):
                                        write_log(player.PLAYER_NAME+' was the best rebounder of '+str(player.TEAM_CITY), log_file)
                                    elif int(player.REB) == int(game.teams[team].TEAM_HIGHEST_REB_2nd):
                                        write_log(player.PLAYER_NAME+' was the second best rebounder of '+str(player.TEAM_CITY), log_file)
                                    if int(player.REB) == int(game.HIGHEST_REB):
                                        write_log(player.PLAYER_NAME+' was the best rebounder in the game.', log_file)
                                    elif int(player.REB) == int(game.HIGHEST_REB_2nd):
                                        write_log(player.PLAYER_NAME+' was the second best rebounder in the game.', log_file)
                            if player.START_POSITION != 'N/A':
                                write_log(player.PLAYER_NAME+' started the game as '+str(player.START_POSITION)+'.', log_file)
                            if player.STL != 'N/A':
                                write_log(player.PLAYER_NAME+' provided '+str(player.STL)+' steals.', log_file)
                                if int(player.STL) > 0:
                                    if int(player.STL) == int(game.teams[team].TEAM_HIGHEST_STL):
                                        write_log(player.PLAYER_NAME+' was the player with most steals of '+str(player.TEAM_CITY), log_file)
                                    # elif int(player.STL) == int(game.teams[team].TEAM_HIGHEST_STL_2nd):
                                        # write_log(player.PLAYER_NAME+' was the second player with most steals of '+str(player.TEAM_CITY), log_file)
                                    if int(player.STL) == int(game.HIGHEST_STL):
                                        write_log(player.PLAYER_NAME+' was the player with most steals in the game.', log_file)
                                    # elif int(player.STL) == int(game.HIGHEST_STL_2nd):
                                        # write_log(player.PLAYER_NAME+' was the second player with most steals in the game.', log_file)
                            if player.TEAM_CITY != 'N/A':
                                write_log(player.PLAYER_NAME+' plays for '+str(player.TEAM_CITY)+'.', log_file)
                            if player.TO != 'N/A':
                                write_log(player.PLAYER_NAME+' committed '+str(player.TO)+' turnovers.', log_file)
                            if player.DOUBLE_FIGURE == 'yes':
                                write_log(player.PLAYER_NAME+' achieved a double figure.', log_file)
                            if player.DOUBLE_DOUBLE == 'yes':
                                write_log(player.PLAYER_NAME+' recorded a double-double.', log_file)
                            if player.DOUBLE_DOUBLE_NEAR == 'yes':
                                write_log(player.PLAYER_NAME+' nearly recorded a double-double.', log_file)
                            if player.TRIPLE_DOUBLE == 'yes':
                                write_log(player.PLAYER_NAME+' recorded a triple-double.', log_file)
                            if player.TRIPLE_DOUBLE_NEAR == 'yes':
                                write_log(player.PLAYER_NAME+' nearly recorded a triple-double.', log_file)
                            if player.GOOD_OFF_BENCH == 'yes':
                                write_log(player.PLAYER_NAME+' was good off-the-bench.', log_file)
                            if player.FG_PCT != 'N/A':
                                if int(player.FG_PCT) > 50:
                                    write_log(player.PLAYER_NAME+' was efficient with the shooting.', log_file)
                                elif int(player.FG_PCT) < 40:
                                    if player.FGA != 'N/A':
                                        if int(player.FGA) > 0:
                                            write_log(player.PLAYER_NAME+' was bad at shooting.', log_file)
                
            count_games += 1
        log_file.close()
        print('--DONE!')
        
        if create_reference_text_file == 'yes':
            y = 0
            fo = codecs.open(filename+'.txt', 'w', 'utf-8')
            fo.write('### File ### '+json_file_path+'\n')
            while y < len(list_games):
                game = list_games[y]
                fo.write(str(y)+':\t')
                fo.write(game.text+'\n')
                y += 1
            fo.close()

if __name__ == "__main__":
    create_json_template(file_list)
