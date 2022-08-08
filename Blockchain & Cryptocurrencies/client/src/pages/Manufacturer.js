import React from "react";
import TextField from "@material-ui/core/TextField";
import Button from "@material-ui/core/Button";
import { useRole } from "../context/RoleContext";
import Navbar from "../components/Navbar";
import { useStyles } from "../components/Styles";
import Grid from "@material-ui/core/Grid";
import Loader from "../components/Loader";

export default function Manufacturer(props) {
    const supplyChainContract = props.supplyChainContract;
    const classes = useStyles();
    const { roles } = useRole();
    const [loading, setLoading] = React.useState(false);
    const [fvalid, setfvalid] = React.useState(false);
    const navItem = [
        ["Buy material", "/manufacturer"],
        ["Process product", "/manufacturer"],
        ["Pack product", "/manufacturer"]
    ];
    const [buyMatForm, setBuyMatForm] = React.useState({
        lotNumber: ""
    });

    const [processForm, setProcessForm] = React.useState({
        upc : 0,
        lotNumbers: "[xxxxxxxx, xxxxxxxx]",
        productName: ""
    });

    const [packProdForm, setPackProdForm] = React.useState({
        upc1: 0,
        productBasePrice: 0
    });

    const handleChangeBuyMatForm = async (e) => {
        setBuyMatForm({
            ...buyMatForm,
            [e.target.name]: e.target.value,
        });
    };

    const handleChangeProcessForm = async (e) => {
        setProcessForm({
            ...processForm,
            [e.target.name]: e.target.value,
        });
    };

    const handleChangePackProdForm = async (e) => {
        setPackProdForm({
            ...packProdForm,
            [e.target.name]: e.target.value,
        });
    };

    const handleSubmitBuyMatForm = async () => {
        setLoading(true);
        
        if (buyMatForm.lotNumber !== "") {
            setfvalid(false);
            await supplyChainContract.methods.buyMaterial(parseInt(buyMatForm.lotNumber)).send({ from: roles.manufacturer, gas: 999999, value: 10 })
                .then(console.log)
                setBuyMatForm({
                    lotNumber: ""
                })
        } else {
            setfvalid(true);
        }
        setLoading(false);
    };

    const handleSubmitProcessForm = async () => {
        setLoading(true);
        
        if (processForm.upc !== "" && processForm.lotNumbers !== "" && processForm.productName !== "" ) {
            setfvalid(false);
            await supplyChainContract.methods.processProduct(parseInt(processForm.upc), processForm.lotNumbers.match(/\d+/g).map(Number), processForm.productName).send({ from: roles.manufacturer, gas: 999999 })
                .then(console.log)
                setProcessForm({
                    upc: "",
                    lotNumbers: "",
                    materialName: ""
                })
        } else {
            setfvalid(true);
        }
        setLoading(false);
    };

    const handleSubmitPackProdForm = async () => {
        setLoading(true);
        
        if (packProdForm.farmerName !== "" && packProdForm.farmerLatitude !== "" ) {
            setfvalid(false);
            await supplyChainContract.methods.packProduct(parseInt(packProdForm.upc1), parseInt(packProdForm.productBasePrice)).send({ from: roles.manufacturer, gas: 999999 })
                .then(console.log)
                setPackProdForm({
                    upc1: 0,
                    productBasePrice: 0
                })
        } else {
            setfvalid(true);
        }
        setLoading(false);
    };

    return (
        <>
            <Navbar pageTitle={"Manufacturer"} navItems={navItem}>
                {loading ? (
                    <Loader />
                ) : (
                    <>
                        <div className={classes.FormWrap}>
                            <h1 className={classes.pageHeading}>Buy Material</h1>
                            <Grid container spacing={3}>
                                <Grid item xs={12}>
                                    <TextField
                                        required
                                        name="lotNumber"
                                        variant="outlined"
                                        value={buyMatForm.lotNumber}
                                        onChange={handleChangeBuyMatForm}
                                        label="Lot Number"
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
                                onClick={handleSubmitBuyMatForm}
                            >
                                BUY
                            </Button>

                            <br />
                            <br />


                        </div>
                        <div className={classes.FormWrap}>
                            <h1 className={classes.pageHeading}>Process Product</h1>
                            <Grid container spacing={3}>
                                <Grid item xs={4}>
                                    <TextField
                                        required
                                        name="upc"
                                        variant="outlined"
                                        value={processForm.upc}
                                        onChange={handleChangeProcessForm}
                                        label="UPC"
                                        style={{ width: "100%" }}
                                    />
                                </Grid>
                                <Grid item xs={4}>
                                    <TextField
                                        required
                                        name="lotNumbers"
                                        variant="outlined"
                                        value={processForm.lotNumbers}
                                        onChange={handleChangeProcessForm}
                                        label="Lot numbers"
                                        style={{ width: "100%" }}
                                    />
                                </Grid>
                                <Grid item xs={4}>
                                    <TextField
                                        required
                                        name="productName"
                                        variant="outlined"
                                        value={processForm.productName}
                                        onChange={handleChangeProcessForm}
                                        label="Product Name"
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
                                onClick={handleSubmitProcessForm}
                            >
                                PROCESS
                    </Button>
                    </div>
                    <div className={classes.FormWrap}>
                            <h1 className={classes.pageHeading}>Pack Product</h1>
                            <Grid container spacing={3}>
                                <Grid item xs={6}>
                                    <TextField
                                        required
                                        name="upc1"
                                        variant="outlined"
                                        value={packProdForm.upc1}
                                        onChange={handleChangePackProdForm}
                                        label="UPC"
                                        style={{ width: "100%" }}
                                    />
                                </Grid>
                                <Grid item xs={6}>
                                    <TextField
                                        required
                                        name="productBasePrice"
                                        variant="outlined"
                                        value={packProdForm.productBasePrice}
                                        onChange={handleChangePackProdForm}
                                        label="Product Price"
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
                                onClick={handleSubmitPackProdForm}
                            >
                                PACK
                    </Button>
                    </div>
                    </>
                )}
            </Navbar>
        </>
    );
}