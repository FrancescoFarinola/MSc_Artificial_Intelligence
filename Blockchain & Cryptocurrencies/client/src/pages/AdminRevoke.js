import React from 'react';
import ResponsiveDrawer from "../components/Navbar";
import TextField from "@material-ui/core/TextField";
import Button from "@material-ui/core/Button";
import { useRole } from "../context/RoleContext";
import { useStyles } from "../components/Styles";

function RoleRevoke(props) {
  const accounts = props.accounts;
  const supplyChainContract = props.supplyChainContract;
  const { roles, setRoles } = useRole();

  const classes = useStyles();
  const [farmerRole, setFarmerRole] = React.useState("");
  const [manufacturerRole, setManufacturerRole] = React.useState("");
  const [distributorRole, setDistributorRole] = React.useState("");
  const [retailerRole, setRetailerRole] = React.useState("");
  const [customerRole, setCustomerRole] = React.useState("");
  const navItem = [];

  const handleRevokeFarmerRole = async () => {
    await setRoles({
      ...roles, 
      farmer : farmerRole
    })

    localStorage.setItem("fRole", null);
    await supplyChainContract.methods.revokeFarmerRole(farmerRole).send({ from: accounts[0], gas:100000 })
    .then(console.log);
    setFarmerRole("");
  }

  const handleRevokeManufacturerRole = async () => {
    await setRoles({
      ...roles, 
      manufacturer : manufacturerRole
    })

    localStorage.setItem("mRole", null);
    await supplyChainContract.methods.revokeManufacturerRole(manufacturerRole).send({ from: accounts[0], gas:100000 })
    .then(console.log);
    setManufacturerRole("");
  }
  
  const handleRevokeDistributorRole = async () => {
    await setRoles({
      ...roles, 
      distributor : distributorRole
    })

    localStorage.setItem("dRole", null);
    await supplyChainContract.methods.revokeDistributorRole(distributorRole).send({ from: accounts[0], gas:100000 })
    .then(console.log);
    setDistributorRole("");
  }

  const handleRevokeRetailerRole = async () => {
    await setRoles({
      ...roles, 
      retailer : retailerRole
  })

   localStorage.setItem("rRole", null);
    await supplyChainContract.methods.revokeRetailerRole(retailerRole).send({ from: accounts[0], gas:100000 })
    .then(console.log);
    setRetailerRole("");
  }

  const handleRevokeCustomerRole = async () => {
    await setRoles({
      ...roles, 
    customer : customerRole
  })

   localStorage.setItem("cRole", null);
    await supplyChainContract.methods.revokeCustomerRole(customerRole).send({ from: accounts[0], gas:100000 })
    .then(console.log);
    setCustomerRole("");
  }


  return (
    <div>
      <ResponsiveDrawer navItems={navItem}>
      <div className={classes.FormWrap}>
      <h1 className={classes.pageHeading}>Revoke Roles</h1>
      {console.log(roles)}

      <form className={classes.root} noValidate autoComplete="off">
        <div className={classes.RoleForm} >
          <TextField
            id="farmerRole"
            label="Enter Farmer address"
            variant="outlined"
            value={farmerRole}
            onChange={(e) => setFarmerRole(e.target.value)}
            style={{width:"70%"}}
          />
          <Button
            variant="contained"
            color="primary"
            onClick={handleRevokeFarmerRole}
            style={{width:"30%", marginLeft:"10px"}}
          >
            Revoke Farmer
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
            onClick={handleRevokeManufacturerRole}
            style={{width:"30%", marginLeft:"10px"}}
          >
            Revoke Manufacturer
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
            onClick={handleRevokeDistributorRole}
            style={{width:"30%", marginLeft:"10px"}}
          >
            Revoke Distributor
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
            onClick={handleRevokeRetailerRole}
            style={{width:"30%", marginLeft:"10px"}}
          >
            Revoke Retailer
          </Button>
        </div>
      </form>

      <form className={classes.root} noValidate autoComplete="off">
        <div className={classes.RoleForm} >
          <TextField
            id="customerRole"
            label=" Enter Customer Address"
            variant="outlined"
            value={customerRole}
            onChange={(e) => setCustomerRole(e.target.value)}
            style={{width:"70%"}}
          />
          <Button
            variant="contained"
            color="primary"
            onClick={handleRevokeCustomerRole}
            style={{width:"30%", marginLeft:"10px"}}
          >
            Revoke Customer
          </Button>
        </div>
      </form>
      </div>

      </ResponsiveDrawer>
    </div>
  );
}

export default RoleRevoke;