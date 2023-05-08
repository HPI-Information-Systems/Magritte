#!/bin/bash
# csv.tar.gz password: magritteResearchHPI

curl -u "xDsdGGGGwNZCLt3:magritteResearchHPI" -H 'X-Requested-With: XMLHttpRequest' 'https://nextcloud.hpi.de/public.php/webdav/' -o predev_instances.tar.gz
tar -xvf predev_instances.tar.gz predev_rowpair/

