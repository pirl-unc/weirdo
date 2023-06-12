set -e

CURR_DIR=`pwd`
LOCAL_DIR=$1

if [[ -z "$LOCAL_DIR" ]]; then
  echo "Directory not supplied, using $CURR_DIR/data/swissprot-dat-files";
  LOCAL_DIR=$CURR_DIR/data/swissprot-dat-files
fi

mkdir -p $LOCAL_DIR

REMOTE_DIR="https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/taxonomic_divisions"

set -x

wget -P $LOCAL_DIR $REMOTE_DIR/uniprot_sprot_{archaea,bacteria,fungi,human,invertebrates,mammals,plants,rodents,vertebrates,viruses}.dat.gz

