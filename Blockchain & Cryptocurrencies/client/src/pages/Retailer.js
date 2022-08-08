import React from "react";
import TextField from "@material-ui/core/TextField";
import Button from "@material-ui/core/Button";
import { useRole } from "../context/RoleContext";
import Navbar from "../components/Navbar";
import { useStyles } from "../components/Styles";
import Grid from "@material-ui/core/Grid";
import Loader from "../components/Loader";

export default function Retailer(props) {
    const supplyChainContract = props.supplyChainContract;
    const classes = useStyles();
    const { roles } = useRole();
    const [loading, setLoading] = React.useState(false);
    const [fvalid, setfvalid] = React.useState(false);
    const navItem = [
        ["Receive product", "/retailer"],
        ["Place product", "/retailer"],
    ];
    const [receiveProductForm, setReceiveProductForm] = React.useState({
        upc : 0
    });

    const [placeProductForm, setPlaceProductForm] = React.useState({
        upc1 : 0,
        productFinalPrice: 0
    });

    const handleChangeReceiveProductForm = async (e) => {
        setReceiveProductForm({
            ...receiveProductForm,
            [e.target.name]: e.target.value,
        });
    };

    const handleChangePlaceProductForm = async (e) => {
        setPlaceProductForm({
            ...placeProductForm,
            [e.target.name]: e.target.value,
        });
    };

    const handleSubmitReceiveProductForm = async () => {
        setLoading(true);
        
        if (receiveProductForm.upc !== "" ) {
            setfvalid(false);
            await supplyChainContract.methods.receiveProduct(parseInt(receiveProductForm.upc)).send({ from: roles.retailer, gas: 999999 })
                .then(console.log)
                setReceiveProductForm({
                    upc: 0
                })
        } else {
            setfvalid(true);
        }
        setLoading(false);
    };

    const handleSubmitPlaceProductForm = async () => {
        setLoading(true);
        
        if (placeProductForm.upc1 !== "" && placeProductForm.productFinalPrice !== "" ) {
            setfvalid(false);
            await supplyChainContract.methods.placeProduct(parseInt(placeProductForm.upc1), parseInt(placeProductForm.productFinalPrice)).send({ from: roles.retailer, gas: 999999 })
                .then(console.log)
                setPlaceProductForm({
                    upc1: 0,
                    productFinalPrice: 0
                })
        } else {
            setfvalid(true);
        }
        setLoading(false);
    };

    return (
        <>
            <Navbar pageTitle={"Retailer"} navItems={navItem}>
                {loading ? (
                    <Loader />
                ) : (
                    <>
                        <div className={classes.FormWrap}>
                            <h1 className={classes.pageHeading}>Receive product</h1>
                            <Grid container spacing={3}>
                                <Grid item xs={12}>
                                    <TextField
                                        required
                                        name="upc"
                                        variant="outlined"
                                        value={receiveProductForm.upc}
                                        onChange={handleChangeReceiveProductForm}
                                        label="UPC"
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
                                onClick={handleSubmitReceiveProductForm}
                            >
                                CONFIRM
                            </Button>

                            <br />
                            <br />


                        </div>
                        <div className={classes.FormWrap}>
                            <h1 className={classes.pageHeading}>Place Product</h1>
                            <Grid container spacing={3}>
                                <Grid item xs={6}>
                                    <TextField
                                        required
                                        name="upc1"
                                        variant="outlined"
                                        value={placeProductForm.upc1}
                                        onChange={handleChangePlaceProductForm}
                                        label="UPC"
                                        style={{ width: "100%" }}
                                    />
                                </Grid>
                                <Grid item xs={6}>
                                    <TextField
                                        required
                                        name="productFinalPrice"
                                        variant="outlined"
                                        value={placeProductForm.productFinalPrice}
                                        onChange={handleChangePlaceProductForm}
                                        label="Final Price"
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
                                onClick={handleSubmitPlaceProductForm}
                            >
                                Place
                    </Button>
                    </div>
                    </>
                )}
            </Navbar>
        </>
    );
}