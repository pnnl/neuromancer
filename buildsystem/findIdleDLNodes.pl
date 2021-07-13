
# DL partitions
my @partitions = grep /^dl/, `sinfo`;

# Idle nodes
my @idle_parts = grep /idle/, @partitions;

if (scalar @idle_parts eq 0) {
  # No idle dl parititons... Just choose the shared one.
  print "dl_shared";
} else {
  my @parts = split / /, $idle_parts[0];
  print $parts[0];
}

