import os, sys, time
#Define path to Openeye and OE_LICENSE
from openeye import oechem, oeomega, oequacpac

def configure_omega():
    """ Configure the OpenEye Omega conformer generator """

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

    # TODO: enumerate protonation states

    return omega


def charge_mol(mol, omega):
    """Generate conformers and assign AM1BCCELF10 charges to mol"""
    if omega(mol):
        #black box function to assign charges to a molecule 
        oequacpac.OEAssignCharges(mol, oequacpac.OEAM1BCCELF10Charges())
    else:
        print("Failed to generate conformation(s) for molecule %s" % mol.GetTitle())
    

def inspect_partial_charges(mol):
    """Returning the sum of formal charges and sum of partial charges"""

    absFCharge = 0
    sumFCharge = 0
    sumPCharge = 0.0
    for atm in mol.GetAtoms():
        sumFCharge += atm.GetFormalCharge()
        absFCharge += abs(atm.GetFormalCharge())
        sumPCharge += atm.GetPartialCharge()
    
    return sumFCharge, sumPCharge

initial_time = time.time()
nmolecules = 0

def update_timer():
    global nmolecules, initial_time
    nmolecules += 1
    total_time = time.time() - initial_time
    average_time = total_time / nmolecules
    print(f'{nmolecules} molecules processed in {total_time} seconds : {average_time} seconds/molecule')


def save_charged_mol(mol, ofs):
    """Write conformer 0 of `mol` into output file stream `ofs`"""
    conf = mol.GetConf(oechem.OEHasConfIdx(0))
    oechem.OEWriteMolecule(ofs, conf)


def parse_input_arguments(argv):
    """Form OE input- and output- file streams
    
    <infile> <outfile> <n_jobs> <job_id>


    modified to handle two more parameters:
    instead of canonical_charges.py <infile> <outfile>
    we want canonical_charges.py <infile> <outfile> <n_batches> <job_id>


    Parameters
    ----------
    argv

    Returns
    -------


    """
    #unnamed argument-a bit lazy (UNIX-comman followed by arguments, or key value pair arguments, recursive, or specify directory)

    # TODO: can use argparse or another parsing utility here
    if len(argv) != 5:
        oechem.OEThrow.Usage("%s <infile> <outfile> <n_jobs> <job_id>" % argv[0])
    
    #input string for openeye 
    input_filename = argv[1]
    output_filename = argv[2]
    n_jobs = int(argv[3])
    job_id = int(argv[4])

    ifs = oechem.oemolistream()
    if not ifs.open(input_filename):
        oechem.OEThrow.Fatal("Unable to open %s for reading" % argv[1])

    # modify output file stream name to reference the job id
    # output.mol2 --> output_1.mol2 (if job_id=1)
    # 'output.mol2'.split('.') --> ['output', 'mol2'] --> 'output_{job_id}.mol2'
    ofs = oechem.oemolostream()
    name, extension = output_filename.split('.')
    job_specific_outname = f'{name}_{job_id}.{extension}'
    print('output filename ', job_specific_outname)

    if not ofs.open(job_specific_outname):
        oechem.OEThrow.Fatal("Unable to open %s for writing" % job_specific_outname)
    
    if ofs.GetFormat() not in [oechem.OEFormat_MOL2, oechem.OEFormat_OEB]:
        oechem.OEThrow.Error("MOL2 or OEB output file is required!")

    return ifs, ofs, n_jobs, job_id


def compute_mol_ids(job_id, n_jobs, total_n_molecules):
    """ assume job_id is 0-indexed """

    # construct batch indices for all jobs
    batch_size = int(total_n_molecules / n_jobs)
    batches = []
    current_batch = []
    for i in range(total_n_molecules):
        if (len(current_batch) >= batch_size) or (i == (total_n_molecules - 1)):
            batches.append(current_batch)
            current_batch = []
        else: 
            current_batch.append(i)

    # get the batch indices for the current job
    this_job = batches[job_id]
    
    return this_job


def main(argv=[__name__]):
    # setup
    omega = configure_omega()

    # let's break this into many jobs, and submit a job array
    ifs, ofs, n_jobs, job_id = parse_input_arguments(argv)

    # note: checking the number of molecules consumes the file stream
    total_n_molecules = len(list(ifs.GetOEMols()))
    print('total n molecules', total_n_molecules)
    ifs = parse_input_arguments(argv)[0]

    this_job = compute_mol_ids(job_id, n_jobs, total_n_molecules)
    print('molecules in current job', len(this_job))

    
    def process_mol(mol):
        """charge the molecule, save it to output stream, and update a timer in the terminal"""
        # TODO: more exception-handling error-logging

        charge_mol(mol, omega)
        #inspect_partial_charges(mol)
        save_charged_mol(mol, ofs)
        update_timer()


    for i, mol in enumerate(ifs.GetOEMols()):
        if i in this_job:
            process_mol(mol)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
