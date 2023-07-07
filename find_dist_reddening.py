import sys

from astropy import coordinates as coord
from astropy import units as u
from astroquery.gaia import Gaia
from dustmaps.bayestar import BayestarQuery
from dustmaps.sfd import SFDQuery
from sigfig import round

Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"

try:
    coo = coord.SkyCoord.from_name(sys.argv[1])
except Exception:
    coo = coord.SkyCoord.from_name(sys.argv[1], parse=True)

r = Gaia.query_object_async(coordinate=coo, width=10 * u.arcsec, height=10 * u.arcsec)
d = coord.Distance(parallax=float(r["parallax"]) * u.mas)
coo = coord.SkyCoord(coo.ra, coo.dec, distance=d)

bayestar = BayestarQuery()
sfd = SFDQuery()

ebv_bayestar = bayestar(coo, mode="percentile", pct=[16.0, 50.0, 84.0])
ebv_sfd = sfd(coo)

pstring = round(float(r["parallax"]), float(r["parallax_error"]))
print(f"Gaia Parallax = {pstring}")

print(f"Bayestar 16th-84th percentiles = {ebv_bayestar}")
print(f"Max E(B-V) from SFD = {ebv_sfd}")
