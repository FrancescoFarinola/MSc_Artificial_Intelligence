import React from "react";
import TextField from "@material-ui/core/TextField";
import Button from "@material-ui/core/Button";
import { useRole } from "../context/RoleContext";
import Navbar from "../components/Navbar";
import { useStyles } from "../components/Styles";
import Grid from "@material-ui/core/Grid";
import Loader from "../components/Loader";

export default function Distributor(props) {
    const supplyChainContract = props.supplyChainContract;
    const classes = useStyles();
    const { roles } = useRole();
    const [loading, setLoading] = React.useState(false);
    const [fvalid, setfvalid] = React.useState(false);
    const navItem = [
        ["Buy product", "/distributor"],
        ["Ship product", "/distributor"],
    ];
    const [buyProductForm, setBuyProductForm] = React.useState({
        upc : 0
    });

    const [shipProductForm, setShipProductForm] = React.useState({
        upc1 : 0
    });

    const handleChangeBuyProductForm = async (e) => {
        setBuyProductForm({
            ...buyProductForm,
            [e.target.name]: e.target.value,
        });
    };

    const handleChangePackForm = async (e) => {
        setShipProductForm({
            ...shipProductForm,
            [e.target.name]: e.target.value,
        });
    };

    const handleSubmitBuyProductForm = async () => {
        setLoading(true);
        
        if (buyProductForm.upc !== "" ) {
            setfvalid(false);
            await supplyChainContract.methods.buyProduct(parseInt(buyProductForm.upc)).send({ from: roles.distributor, value: 10, gas: 999999 })
                .then(console.log)
                setBuyProductForm({
                    upc: 0
                })
        } else {
            setfvalid(true);
        }
        setLoading(false);
    };

    const handleSubmitShipProductForm = async () => {
        setLoading(true);
        
        if (shipProductForm.upc1 !== "" ) {
            setfvalid(false);
            await supplyChainContract.methods.shipProduct(parseInt(shipProductForm.upc1)).send({ from: roles.distributor, gas: 999999 })
                .then(console.log)
                setShipProductForm({
                    upc1: 0
                })
        } else {
            setfvalid(true);
        }
        setLoading(false);
    };

    return (
        <>
            <Navbar pageTitle={"Distributor"} navItems={navItem}>
                {loading ? (
                    <Loader />
                ) : (
                    <>
                        <div className={classes.FormWrap}>
                            <h1 className={classes.pageHeading}>Buy product</h1>
                            <Grid container spacing={3}>
                                <Grid item xs={12}>
                                    <TextField
                                        required
                                        name="upc"
                                        variant="outlined"
                                        value={buyProductForm.upc}
                                        onChange={handleChangeBuyProductForm}
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
                                onClick={handleSubmitBuyProductForm}
                            >
                                BUY
                            </Button>

                            <br />
                            <br />


                        </div>
                        <div className={classes.FormWrap}>
                            <h1 className={classes.pageHeading}>Pack Product</h1>
                            <Grid container spacing={3}>
                                <Grid item xs={12}>
                                    <TextField
                                        required
                                        name="upc1"
                                        variant="outlined"
                                        value={shipProductForm.upc1}
                                        onChange={handleChangePackForm}
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
                                onClick={handleSubmitShipProductForm}
                            >
                                SHIP
                    </Button>
                    </div>
                    </>
                )}
            </Navbar>
        </>
    );
}