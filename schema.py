"""Feature order from KDD Cup 99 and the categories used in its training set."""
FEATURES = """duration protocol_type service flag src_bytes dst_bytes land wrong_fragment
urgent hot num_failed_logins logged_in num_compromised root_shell su_attempted num_root
num_file_creations num_shells num_access_files num_outbound_cmds is_host_login is_guest_login
count srv_count serror_rate srv_serror_rate rerror_rate srv_rerror_rate same_srv_rate
diff_srv_rate srv_diff_host_rate dst_host_count dst_host_srv_count dst_host_same_srv_rate
dst_host_diff_srv_rate dst_host_same_src_port_rate dst_host_srv_diff_host_rate
dst_host_serror_rate dst_host_srv_serror_rate dst_host_rerror_rate dst_host_srv_rerror_rate""".split()

FAMILIES = {
    "normal": "normal",
    "dos": "back land neptune pod smurf teardrop",
    "probe": "ipsweep nmap portsweep satan",
    "r2l": "ftp_write guess_passwd imap multihop phf spy warezclient warezmaster",
    "u2r": "buffer_overflow loadmodule perl rootkit",
}
LABELS = {label: family for family, labels in FAMILIES.items() for label in labels.split()}
DROP = set("""num_root srv_serror_rate srv_rerror_rate dst_host_srv_serror_rate
dst_host_serror_rate dst_host_rerror_rate dst_host_srv_rerror_rate dst_host_same_srv_rate
is_host_login num_outbound_cmds service""".split())
CATEGORIES = {"protocol_type": "icmp tcp udp".split(),
              "flag": "SF S0 REJ RSTR RSTO SH S1 S2 RSTOS0 S3 OTH".split()}
