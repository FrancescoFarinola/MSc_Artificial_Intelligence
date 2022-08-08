import React from 'react';
import ResponsiveDrawer from "../components/Navbar";
import TextField from "@material-ui/core/TextField";
import Button from "@material-ui/core/Button";
import { useRole } from "../context/RoleContext";
import { useStyles } from "../components/Styles";

function RoleAdd(props) {
  const accounts = props.accounts;
  const supplyChainContract = props.supplyChainContract;
  const { roles, setRoles } = useRole();

  const classes = useStyles();
  const [farmerRole, setFarmerRole] = React.useState("");
  const [manufacturerRole, setManufacturerRole] = React.useState("");
  const [distributorRole, setDistributorRole] = React.useState("");
  const [retailerRole, setRetailerRole] = React.useState("");
  const [consumerRole, setConsumerRole] = React.useState("");
  const navItem = [];

  const handleAddFarmerRole = async () => {
    await setRoles({
      ...roles, 
      farmer : farmerRole
    })

    localStorage.setItem("fRole", farmerRole);
    await supplyChainContract.methods.addFarmerRole(farmerRole).send({ from: accounts[0], gas:100000 })
    .then(console.log);
    setFarmerRole("");
  }

  const handleAddManufacturerRole = async () => {
    await setRoles({
      ...roles, 
      manufacturer : manufacturerRole
    })

    localStorage.setItem("mRole", manufacturerRole);
    await supplyChainContract.methods.addManufacturerRole(manufacturerRole).send({ from: accounts[0], gas:100000 })
    .then(console.log);
    setManufacturerRole("");
  }
  
  const handleAddDistributorRole = async () => {
    await setRoles({
      ...roles, 
      distributor : distributorRole
    })

    localStorage.setItem("dRole", distributorRole);
    await supplyChainContract.methods.addDistributorRole(distributorRole).send({ from: accounts[0], gas:100000 })
    .then(console.log);
    setDistributorRole("");
  }

  const handleAddRetailerRole = async () => {
    await setRoles({
      ...roles, 
      retailer : retailerRole
  })

   localStorage.setItem("rRole", retailerRole);
    await supplyChainContract.methods.addRetailerRole(retailerRole).send({ from: accounts[0], gas:100000 })
    .then(console.log);
    setRetailerRole("");
  }

  const handleAddConsumerRole = async () => {
    await setRoles({
      ...roles, 
    consumer : consumerRole
  })

   localStorage.setItem("cRole", consumerRole);
    await supplyChainContract.methods.addConsumerRole(consumerRole).send({ from: accounts[0], gas:100000 })
    .then(console.log);
    setConsumerRole("");
  }


  return (
    <div>
      <ResponsiveDrawer navItems={navItem}>
      <div className={classes.FormWrap}>
      <h1 className={classes.pageHeading}>Add Roles</h1>
      {console.log(roles)}

      <form className={classes.root} noValidate autoComplete="off">
        <div className={classes.RoleForm} >
          <TextField
            id="farmerRole"
            label="Enter Farmer Address"
            variant="outlined"
            value={farmerRole}
            onChange={(e) => setFarmerRole(e.target.value)}
            style={{width:"70%"}}
          />
          <Button
            variant="contained"
            color="primary"
            onClick={handleAddFarmerRole}
            style={{width:"30%", marginLeft:"10px"}}
          >
            Add Farmer
          </Button>
        </div>
      </form>
      
      <form className={classes.root} noValidate autoComplete="off">
        <div className={classes.RoleForm} >
          <TextField
            id="manufacturerRole"
            label="Enter Manufacturer Address"
            variant="outlined"
            value={manufacturerRole}
            onChange={(e) => setManufacturerRole(e.target.value)}
            style={{width:"70%"}}
          />
          <Button
            variant="contained"
            color="primary"
            onClick={handleAddManufacturerRole}
            style={{width:"30%", marginLeft:"10px"}}
          >
            Add Manufacturer
          </Button>
        </div>
      </form>

      <form className={classes.root} noValidate autoComplete="off">
        <div className={classes.RoleForm} >
          <TextField
            id="distributorRole"
            label="Enter Distributor Address "
            variant="outlined"
            value={distributorRole}
            onChange={(e) => setDistributorRole(e.target.value)}
            style={{width:"70%"}}
          />
          <Button
            variant="contained"
            color="primary"
            onClick={handleAddDistributorRole}
            style={{width:"30%", marginLeft:"10px"}}
          >
            Add Distributor
          </Button>
        </div>
      </form>

      <form className={classes.root} noValidate autoComplete="off">
        <div className={classes.RoleForm} >
          <TextField
            id="retailerRole"
            label="Enter Retailer Address"
            variant="outlined"
            value={retailerRole}
            onChange={(e) => setRetailerRole(e.target.value)}
            style={{width:"70%"}}
          />
          <Button
            variant="contained"
            color="primary"
            onClick={handleAddRetailerRole}
            style={{width:"30%", marginLeft:"10px"}}
          >
            Add Retailer
          </Button>
        </div>
      </form>

      <form className={classes.root} noValidate autoComplete="off">
        <div className={classes.RoleForm} >
          <TextField
            id="consumerRole"
            label=" Enter Consumer Address"
            variant="outlined"
            value={consumerRole}
            onChange={(e) => setConsumerRole(e.target.value)}
            style={{width:"70%"}}
          />
          <Button
            variant="contained"
            color="primary"
            onClick={handleAddConsumerRole}
            style={{width:"30%", marginLeft:"10px"}}
          >
            Add Consumer
          </Button>
        </div>
      </form>
      </div>


      </ResponsiveDrawer>
    </div>
  );
}

export default RoleAdd;