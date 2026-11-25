"""Every screen and element id the code uses, in one place (B1).

The names come from ``assets/ui_v3/manifest.json`` (hotspot ``n`` and screen
names) plus a few ids the code itself adds (modal buttons, forgot-password
fields). Code never spells an id as a string: it uses these constants, and
``tests/test_ui_ids.py`` / ``tests/test_architecture.py`` keep it so.
"""

from __future__ import annotations

# ---- screens
SIGN_IN = 'v3_01a_sign_in'
SIGN_UP = 'v3_01b_sign_up'
SIGN_IN_ERR = 'v3_01c_sign_in_error'
MENU = 'v3_02a_menu'
SETTINGS = 'v3_02b_settings'
OFFLINE = 'v3_02c_disconnected'
LEVELS = 'v3_03_level_select'
PHOTOS = 'v3_04_choose_photo'
LOADOUT = 'v3_05_loadout'
TILT = 'v3_06_head_tilt_check'
LOADING = 'v3_07_loading'
LOADING_READY = 'v3_07_loading_ready'
PLAY = 'v3_08_gameplay_hud'
WIN = 'v3_09a_result_win'
LOSE = 'v3_09b_result_lose'
PVP_MODE = 'v3_10_pvp_mode'
PVP_SEARCH = 'v3_11a_pvp_searching'
PVP_INVITE = 'v3_11b_pvp_invite'
PVP_FOUND = 'v3_11c_pvp_found'
PVP_HUNT = 'v3_12a_pvp_hunter_hud'
PVP_HIDE = 'v3_12b_pvp_bug_hud'
PVP_RESULT = 'v3_13_pvp_result'
BOARD = 'v3_14_leaderboard'

# ---- elements (hotspots and ids the code adds)
FIELD_USER = 'field_user'
ICON_SHOW_PASSWORD = 'icon_show_password'
FIELD_PASS = 'field_pass'
SWITCH_REMEMBER = 'switch_remember'
LINK_FORGOT = 'link_forgot'
BTN_SIGN_IN = 'btn_Sign in'
BTN_CREATE_ALBUM = 'btn_Create album'
LINK_GUEST = 'link_guest'
FIELD_PASS2 = 'field_pass2'
FOOTER_LINK = 'footer_link'
BTN_CREATE_A_NEW_ALBUM = 'btn_Create a new album'
BTN_SOLO = 'btn_Solo'
BTN_PVP = 'btn_PvP'
BTN_LEADERBOARD = 'btn_Leaderboard'
BTN_SETTINGS = 'btn_Settings'
SIGN_OUT = 'sign_out'
THEME_DAY = 'theme_day'
THEME_NIGHT = 'theme_night'
DISPLAY_FULL = 'display_full'
DISPLAY_WINDOW = 'display_window'
DROPDOWN_WEBCAM = 'dropdown_webcam'
BTN_RECALIBRATE_LOWER = 'btn_recalibrate'
SLIDER = 'slider'
BTN_SIGN_OUT = 'btn_Sign out'
TAB_SOLO = 'tab_Solo'
TAB_PVP = 'tab_PvP'
TAB_LEADERBOARD = 'tab_Leaderboard'
TAB_SETTINGS = 'tab_Settings'
BTN_TRY_AGAIN_NOW = 'btn_Try again now'
BTN_BACK_TO_ALBUM = 'btn_Back to album'
LEVEL_ROW_2 = 'level_row_2'
BTN_BACK = 'btn_Back'
BTN_NEXT_CHOOSE_A_PHOTO = 'btn_Next: choose a photo'
BTN_CHOOSE_FROM_COMPUTER = 'btn_Choose from computer'
BTN_SPLIT_LAYERS_CONTINUE = 'btn_Split layers & continue'
BTN_SHARE_TO_ALBUM = 'btn_Share to album'
BTN_RANDOM_FROM_ALBUM = 'btn_Random from album'
DROPZONE = 'dropzone'
RECENT_MEMORY = 'recent_memory'
BTN_START = 'btn_Start'
BTN_RECALIBRATE = 'btn_Recalibrate'
BTN_ENTER_MATCH = 'btn_Enter match'
CONTINUE = 'continue'
BTN_NEXT_LEVEL = 'btn_Next level'
BTN_PLAY_AGAIN = 'btn_Play again'
BTN_TRY_AGAIN = 'btn_Try again'
BTN_CHOOSE_ANOTHER_PHOTO = 'btn_Choose another photo'
BTN_INVITE_A_FRIEND = 'btn_Invite a friend'
BTN_FIND_MATCH = 'btn_Find match'
BTN_CANCEL = 'btn_Cancel'
FIELD_JOIN = 'field_join'
BTN_COPY_CODE = 'btn_Copy code'
BTN_JOIN_ROOM = 'btn_Join room'
BTN_CLOSE_ROOM = 'btn_Close room'
LEAVE = 'leave'
BTN_REMATCH = 'btn_Rematch'
BTN_CONFIRM_NO = 'btn_confirm_no'
BTN_CONFIRM_YES = 'btn_confirm_yes'
BTN_FORGOT_CANCEL = 'btn_forgot_cancel'
BTN_FORGOT_SUBMIT = 'btn_forgot_submit'
BTN_RECOVERY_COPY = 'btn_recovery_copy'
BTN_RECOVERY_SAVED = 'btn_recovery_saved'
FIELD_FORGOT_CODE = 'field_forgot_code'
FIELD_FORGOT_PASS = 'field_forgot_pass'
FIELD_FORGOT_PASS2 = 'field_forgot_pass2'
FIELD_FORGOT_USER = 'field_forgot_user'

#: Manifest ids that no code spells out, because they are built from a
#: prefix plus an item or a number (``"sticker_" + item``, ``"level_row_%d"``).
NOT_USED_IN_CODE = frozenset({
    'level_row_1',
    'level_row_3',
    'level_row_4',
    'level_row_5',
    'weapon_Precision gun',
    'weapon_Spread gun',
    'weapon_Heavy-mag gun',
    'sticker_Flare',
    'sticker_Antidote',
    'sticker_Acid shield',
    'sticker_Filter lens',
    'sticker_Compass',
    'sticker_Thermal scan',
    'sticker_Energy drink',
    'sticker_Smoke bomb',
    'slot_empty_1',
    'slot_empty_2',
    'slot_empty_3',
    'mode_Competitive',
    'mode_Hunt & Hide',
    'seg_Solo',
    'seg_PvP · Competitive',
    'seg_PvP · Hunt & Hide',
})
