import React, { Component } from "react";
import { Router, Switch, Route } from "react-router-dom";
import SupplyChainContract from "./contracts/SupplyChain.json";
import getWeb3 from "./getWeb3";

import { ThemeProvider } from "@material-ui/core/styles";
import theme from "./components/Theme";
import {createBrowserHistory} from 'history';

import { RoleContextProvider } from "./context/RoleContext";
import RoleAdd from "./pages/AdminAdd";
import RoleRevoke from "./pages/AdminRevoke";
import Explorer from './pages/Explorer';
import Home from "./pages/Home";

import Farmer from "./pages/Farmer";
import Manufacturer from "./pages/Manufacturer";
import Distributor from "./pages/Distributor";
import Retailer from "./pages/Retailer";
import Consumer from "./pages/Consumer";



class App extends Component {

  state = { web3: null, accounts: null, contract: null, fRole: null, mRole: null, dRole: null, rRole: null, cRole: null };

  constructor(props) {
    super(props);

    this.handleAccountsChanged = this.handleAccountsChanged.bind(this);
    this.handleChainChanged = this.handleChainChanged.bind(this)

    this.toast = React.createRef();
  }

  handleChainChanged(_chainId) {
    // Just reload page
    window.location.reload();
  }

  handleAccountsChanged(accounts) {
    if (this.state) {
      console.log("Accounts changed", accounts, this.state.currentAccount)
      if (accounts.length === 0) {
        // MetaMask is locked or the user has not connected any accounts
        this.showError('Please connect to MetaMask.');
      } else if (accounts[0] !== this.state.currentAccount) {
        this.setState({ currentAccount: accounts[0] })
      }
    } else {
      console.log("React timing issue", this, this.state)
    }
  }

  componentDidMount = async () => {
    try {
      // Get network provider and web3 instance.
      const { web3, ethereum, accounts } = await getWeb3();

      // Get the contract instance.
      const networkId = await web3.eth.net.getId();
      console.log(networkId)
      console.log(accounts)

      ethereum.on('chainChanged', this.handleChainChanged);
      ethereum.on('accountsChanged', this.handleAccountsChanged);

      const deployedNetwork = SupplyChainContract.networks[networkId];
      const instance = new web3.eth.Contract(
        SupplyChainContract.abi,
        deployedNetwork && deployedNetwork.address,
      );

      const fRole = localStorage.getItem("fRole");
      const mRole = localStorage.getItem("mRole");
      const dRole = localStorage.getItem("dRole");
      const rRole = localStorage.getItem("rRole");
      const cRole = localStorage.getItem("cRole");

      this.setState({ web3, accounts, currentAccount: accounts[0], networkId, contract: instance, fRole: fRole, mRole: mRole, dRole: dRole, rRole: rRole, cRole: cRole }, this.runExample);


    } catch (error) {

      if (typeof error === 'string') {
        this.showPersistentError(
          error,
        );
      } else {
        // Catch any errors for any of the above operations.
        this.showPersistentError(
          `Failed to load web3, accounts, or contract. Check console for details.`,
        );
        console.error(error);
      }

    }
  };


  runExample = async () => {
    const { contract } = this.state;
    console.log(contract);
    console.log(this.state);
  };

  showError = (message) => {
    this.toast.current.show({ severity: 'error', summary: 'Fehler', detail: message, life: 5000 });
  }

  showPersistentError = message => {
    this.toast.current.show({ severity: 'error', summary: 'Fehler', detail: message, sticky: true });
  }

  showInfo = (message) => {
    this.toast.current.show({ severity: 'info', summary: 'Info', detail: message, life: 5000 })
  }

  render() {
    if (!this.state.web3) {
      return <div>Loading Web3, accounts, and contract...</div>;
    }
    return (
      <div className="App">
        <ThemeProvider theme={theme}>
        <RoleContextProvider fRole={this.state.fRole} mRole={this.state.mRole} dRole={this.state.dRole} rRole={this.state.rRole} cRole={this.state.cRole}>
          <Router history={createBrowserHistory()}>
            <Switch>
              <Route exact path="/AdminAdd">
                <RoleAdd accounts={this.state.accounts} supplyChainContract={this.state.contract} />
              </Route>
              <Route exact path="/AdminRevoke">
                <RoleRevoke accounts={this.state.accounts} supplyChainContract={this.state.contract} />
              </Route>
              <Route exact path="/explorer">
                <Explorer accounts={this.state.accounts} supplyChainContract={this.state.contract} web3={this.state.web3} />
              </Route>
              <Route exact path="/">
                <Home accounts={this.state.accounts} supplyChainContract={this.state.contract} />
              </Route>
              <Route exact path="/farmer">
                {this.state.fRole !== "" ? 
                <Farmer accounts={this.state.accounts} supplyChainContract={this.state.contract} />
                : <h1>Assign Farmer Role at /RoleAdmin</h1> }
              </Route>
              <Route exact path="/manufacturer">
                {this.state.mRole !== "" ? 
                <Manufacturer accounts={this.state.accounts} supplyChainContract={this.state.contract} />
                : <h1>Assign Manufacturer Role at /RoleAdmin</h1> }
              </Route>
              <Route exact path="/distributor">
                {this.state.dRole !== "" ? 
                <Distributor accounts={this.state.accounts} supplyChainContract={this.state.contract} />
                : <h1>Assign Distributor Role at /RoleAdmin</h1> }
              </Route>
              <Route exact path="/retailer">
                {this.state.rRole !== "" ? 
                <Retailer accounts={this.state.accounts} supplyChainContract={this.state.contract} />
                : <h1>Assign Retailer Role at /RoleAdmin</h1> }
              </Route>
              <Route exact path="/consumer">
                {this.state.cRole !== "" ? 
                <Consumer accounts={this.state.accounts} supplyChainContract={this.state.contract} />
                : <h1>Assign Consumer Role at /RoleAdmin</h1> }
              </Route>
            </Switch>
          </Router>
        </RoleContextProvider>
        </ThemeProvider>
      </div>
    );
  }
}


export default App;
