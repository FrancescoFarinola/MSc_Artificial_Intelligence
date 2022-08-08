import React from "react";
import TextField from "@material-ui/core/TextField";
import Button from "@material-ui/core/Button";
import { useRole } from "../context/RoleContext";
import Navbar from "../components/Navbar";
import { useStyles } from "../components/Styles";
import Grid from "@material-ui/core/Grid";
import Loader from "../components/Loader";

export default function Farmer(props) {
    const supplyChainContract = props.supplyChainContract;
    const classes = useStyles();
    const { roles } = useRole();
    const [loading, setLoading] = React.useState(false);
    const [fvalid, setfvalid] = React.useState(false);
    const navItem = [
        ["Harvest material", "/farmer"],
        ["Pack material", "/farmer"],
    ];
    const [harvestForm, setHarvestForm] = React.useState({
        farmerName: "",
        farmerLatitude: "",
        farmerLongitude: "",
        materialName: "",
        maxQuantity: 0,
        lotNumber: "-----------"
    });

    const [packForm, setPackForm] = React.useState({
        lotNumber1: "",
        materialQuantity: 0,
        materialPrice: 0
    });

    const handleChangeHarvestForm = async (e) => {
        setHarvestForm({
            ...harvestForm,
            [e.target.name]: e.target.value,
        });
    };

    const handleChangePackForm = async (e) => {
        setPackForm({
            ...packForm,
            [e.target.name]: e.target.value,
        });
    };

    const handleSubmitHarvestForm = async () => {
        setLoading(true);
        
        if (harvestForm.farmerName !== "" && harvestForm.farmerLatitude !== "" && harvestForm.farmerLongitude !== "" && harvestForm.materialName !== "" && harvestForm.maxQuantity) {
            setfvalid(false);
            let result = await supplyChainContract.methods.harvestMaterial(harvestForm.farmerName, harvestForm.farmerLatitude, harvestForm.farmerLongitude, harvestForm.materialName, parseInt(harvestForm.maxQuantity)).send({ from: roles.farmer, gas: 999999 })
                
                setHarvestForm({
                    farmerName: "",
                    farmerLatitude: "",
                    farmerLongitude: "",
                    materialName: "",
                    maxQuantity: 0, 
                    lotNumber: result.events.Harvested.returnValues[0]
                })
            console.log(result.events.Harvested.returnValues[0]);
        } else {
            setfvalid(true);
        }
        setLoading(false);
    };

    const handleSubmitPackForm = async () => {
        setLoading(true);
        
        if (packForm.lotNumber1 !== "" && packForm.materialQuantity !== "" && packForm.materialPrice !== "" ) {
            setfvalid(false);
            await supplyChainContract.methods.packMaterial(parseInt(packForm.lotNumber1), parseInt(packForm.materialQuantity), parseInt(packForm.materialPrice)).send({ from: roles.farmer, gas: 999999 })
                .then(console.log)
                setPackForm({
                    lotNumber1: "",
                    materialQuantity: "",
                    materialPrice: ""
                })
        } else {
            setfvalid(true);
        }
        setLoading(false);
    };

    return (
        <>
            <Navbar pageTitle={"Farmer"} navItems={navItem}>
                {loading ? (
                    <Loader />
                ) : (
                    <>
                        <div className={classes.FormWrap}>
                            <h1 className={classes.pageHeading}>Harvest Material</h1>
                            <Grid container spacing={3}>
                                <Grid item xs={12}>
                                    <TextField
                                        required
                                        name="farmerName"
                                        variant="outlined"
                                        value={harvestForm.farmerName}
                                        onChange={handleChangeHarvestForm}
                                        label="Farmer Name"
                                        style={{ width: "100%" }}
                                    />
                                </Grid>
                                <Grid item xs={6}>
                                    <TextField
                                        required
                                        name="farmerLatitude"
                                        variant="outlined"
                                        value={harvestForm.farmerLatitude}
                                        onChange={handleChangeHarvestForm}
                                        label="Latitude"
                                        style={{ width: "100%" }}
                                    />
                                </Grid>
                                <Grid item xs={6}>
                                    <TextField
                                        required
                                        name="farmerLongitude"
                                        variant="outlined"
                                        value={harvestForm.farmerLongitude}
                                        onChange={handleChangeHarvestForm}
                                        label="Longitude"
                                        style={{ width: "100%" }}
                                    />
                                </Grid>
                                <Grid item xs={6}>
                                    <TextField
                                        required
                                        name="materialName"
                                        variant="outlined"
                                        value={harvestForm.materialName}
                                        onChange={handleChangeHarvestForm}
                                        label="Material Name"
                                        style={{ width: "100%" }}
                                    />
                                </Grid>
                                <Grid item xs={6}>
                                    <TextField
                                        required
                                        name="maxQuantity"
                                        variant="outlined"
                                        value={harvestForm.maxQuantity}
                                        onChange={handleChangeHarvestForm}
                                        label="Quantity Harvested"
                                        style={{ width: "100%" }}
                                    />
                                </Grid>
                                <Grid item xs={6}>
                                    <TextField
                                        readOnly
                                        name="lotNumber"
                                        variant="outlined"
                                        value={harvestForm.lotNumber}
                                        onChange={handleChangeHarvestForm}
                                        label="Lot Number"
                                        style={{ width: "100%" }}
                                        inputProps={
                                            { readOnly: true, }
                                        }
                                    />
                                </Grid>
                            </Grid>
                            <br />
                            <p><b style={{ color: "red" }}>{fvalid ? "Please enter all data" : ""}</b></p>
                            <Button
                                type="submit"
                                variant="contained"
                                color="primary"
                                onClick={handleSubmitHarvestForm}
                            >
                                SUBMIT
                            </Button>

                            <br />
                            <br />


                        </div>
                        <div className={classes.FormWrap}>
                            <h1 className={classes.pageHeading}>Pack Material</h1>
                            <Grid container spacing={3}>
                                <Grid item xs={6}>
                                    <TextField
                                        required
                                        name="lotNumber1"
                                        variant="outlined"
                                        value={packForm.lotNumber1}
                                        onChange={handleChangePackForm}
                                        label="Lot Number"
                                        style={{ width: "100%" }}
                                    />
                                </Grid>
                                <Grid item xs={6}>
                                    <TextField
                                        required
                                        name="materialQuantity"
                                        variant="outlined"
                                        value={packForm.materialQuantity}
                                        onChange={handleChangePackForm}
                                        label="Quantity"
                                        style={{ width: "100%" }}
                                    />
                                </Grid>
                                <Grid item xs={6}>
                                    <TextField
                                        required
                                        name="materialPrice"
                                        variant="outlined"
                                        value={packForm.materialPrice}
                                        onChange={handleChangePackForm}
                                        label="Price"
                                        style={{ width: "100%" }}
                                    />
                                </Grid>
                            </Grid>
                            <br />
                            <p><b style={{ color: "red" }}>{fvalid ? "Please enter all data" : ""}</b></p>
                            <Button
                                type="submit"
                                variant="contained"
                                color="primary"
                                onClick={handleSubmitPackForm}
                            >
                                SUBMIT
                    </Button>
                    </div>
                    </>
                )}
            </Navbar>
        </>
    );
}