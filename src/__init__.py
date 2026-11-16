"""Head-Coupled Perspective Display — open-source HCPD engine."""

__version__ = "0.1.0"

#: Protected M release version. Sent in the PvP handshake (``hello``) so a
#: server and a client built from different commits refuse to play together
#: instead of desyncing silently. TASK_pvp_lan_1116.
VERSION = "1.0.1"
