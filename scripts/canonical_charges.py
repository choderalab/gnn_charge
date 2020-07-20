import os
import sys
import time
#Define path to Openeye and OE_LICENSE
from openeye import oechem
from openeye import oeomega
from openeye import oequacpac

#unnamed argument-a bit lazy (UNIX-comman followed by arguments, or key value pair arguments, recursive, or specify directory)
def main(argv=[__name__]):
    if len(argv) != 3:
        oechem.OEThrow.Usage("%s <infile> <outfile>" % argv[0])
#input string for openeye 
    ifs = oechem.oemolistream()
    if not ifs.open(argv[1]):
        oechem.OEThrow.Fatal("Unable to open %s for reading" % argv[1])
    #if not oechem.OEIs3DFormat(ifs.GetFormat()):
    #    oechem.OEThrow.Fatal("Invalid input format: need 3D coordinates")
    ofs = oechem.oemolostream()
    if not ofs.open(argv[2]):
        oechem.OEThrow.Fatal("Unable to open %s for writing" % argv[2])
    if ofs.GetFormat() not in [oechem.OEFormat_MOL2, oechem.OEFormat_OEB]:
        oechem.OEThrow.Error("MOL2 or OEB output file is required!")
    omega = oeomega.OEOmega()
    #omega.SetIncludeInput(True)
    #include geometry in input and output 
    omega.SetIncludeInput(False)
    omega.SetCanonOrder(False)
    omega.SetSampleHydrogens(True)
    omega.SetStrictStereo(False) # JDC
    #if this is taking too long then throw in an error
    omega.SetMaxSearchTime(2.0) # maximum omega search time
    #energy window= tolerance for high energy conformers (more diverse conformers, less stable)
    eWindow = 15.0
    omega.SetEnergyWindow(eWindow)
    omega.SetMaxConfs(800)
    #RMSThreshold--> wouldn't know when to call two conformers the same or different 
    omega.SetRMSThreshold(1.0)
    initial_time = time.time()
    nmolecules = 0
    for mol in ifs.GetOEMols():
        if omega(mol):
            #black box function to assign charges to a molecule 
            oequacpac.OEAssignCharges(mol, oequacpac.OEAM1BCCELF10Charges())
            conf = mol.GetConf(oechem.OEHasConfIdx(0))
            absFCharge = 0
            sumFCharge = 0
            sumPCharge = 0.0
            for atm in mol.GetAtoms():
                sumFCharge += atm.GetFormalCharge()
                absFCharge += abs(atm.GetFormalCharge())
                sumPCharge += atm.GetPartialCharge()
            print("{}: {} formal charges give total charge {}"
                  "; sum of partial charges {:5.4f}".format(mol.GetTitle(), absFCharge,
                                                            sumFCharge, sumPCharge))
            oechem.OEWriteMolecule(ofs, conf)
            nmolecules += 1
            total_time = time.time() - initial_time
            average_time = total_time / nmolecules
            print(f'{nmolecules} molecules processed in {total_time} seconds : {average_time} seconds/molecule')
        else:
            print("Failed to generate conformation(s) for molecule %s" % mol.GetTitle())
    return 0
if __name__ == "__main__":
    sys.exit(main(sys.argv))
