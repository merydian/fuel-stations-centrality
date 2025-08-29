#!/bin/bash

curl -O "https://download.geofabrik.de/asia/mongolia-latest.osm.pbf"

osmium tags-filter "mongolia-latest.osm.pbf" w/highway=motorway,trunk,primary,secondary,tertiary,residential -o "mongolia-latest_filtered.osm.pbf"

osmium cat "mongolia-latest.osm.pbf" -o "mongolia-latest.osm"

scp "mongolia-latest.osm" helix:ma/fuel-stations-centrality/data/
