#!/bin/bash
# csv.tar.gz password: magritteResearchHPI

# predev_instances.tar.gz https://nextcloud.hpi.de/s/5qXdobmbKqwKGAA password: magritteResearchHPI

curl -u "5qXdobmbKqwKGAA:magritteResearchHPI" -H 'X-Requested-With: XMLHttpRequest' 'https://nextcloud.hpi.de/public.php/webdav/' -o predev_instances.tar.gz
tar -xvf predev_instances.tar.gz predev_rowpair/

