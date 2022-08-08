import React from "react";
import TextField from "@material-ui/core/TextField";
import Button from "@material-ui/core/Button";
import { useRole } from "../context/RoleContext";
import Navbar from "../components/Navbar";
import { useStyles } from "../components/Styles";
import Grid from "@material-ui/core/Grid";
import Loader from "../components/Loader";

export default function Consumer(props) {
    const supplyChainContract = props.supplyChainContract;
    const classes = useStyles();
    const { roles } = useRole();
    const [loading, setLoading] = React.useState(false);
    const [fvalid, setfvalid] = React.useState(false);
    const navItem = [
        ["Purchase product", "/consumer"]
    ];
    const [purchaseProductForm, setPurchaseProductForm] = React.useState({
        upc : 0
    });


    const handleChangePurchaseProductForm = async (e) => {
        setPurchaseProductForm({
            ...purchaseProductForm,
            [e.target.name]: e.target.value,
        });
    };

    const handleSubmitPurchaseProductForm = async () => {
        setLoading(true);
        
        if (purchaseProductForm.upc !== "" ) {
            setfvalid(false);
            await supplyChainContract.methods.purchaseProduct(parseInt(purchaseProductForm.upc)).send({ from: roles.consumer, gas: 999999, value: 10 })
                .then(console.log)
                setPurchaseProductForm({
                    upc: 0
                })
        } else {
            setfvalid(true);
        }
        setLoading(false);
    };

    return (
        <>
            <Navbar pageTitle={"Consumer"} navItems={navItem}>
                {loading ? (
                    <Loader />
                ) : (
                    <>
                        <div className={classes.FormWrap}>
                            <h1 className={classes.pageHeading}>Purchase product</h1>
                            <Grid container spacing={3}>
                                <Grid item xs={12}>
                                    <TextField
                                        required
                                        name="upc"
                                        variant="outlined"
                                        value={purchaseProductForm.upc}
                                        onChange={handleChangePurchaseProductForm}
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
                                onClick={handleSubmitPurchaseProductForm}
                            >
                                PURCHASE
                            </Button>

                            <br />
                            <br />


                        </div>
                    </>
                )}
            </Navbar>
        </>
    );
}